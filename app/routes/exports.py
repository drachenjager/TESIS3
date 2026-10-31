"""Rutas de exportación: Excel y PDF."""

import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")                   # backend sin pantalla (seguro en servidor)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Blueprint, send_file, request
from reportlab.lib import colors as rl_colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    Flowable, HRFlowable, Image as RLImage,
    Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from app import cache
from src.models.ranking import add_ranking

bp = Blueprint("exports", __name__)

matplotlib.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.facecolor":   "white",
    "axes.facecolor":     "#F8FAFD",
})


# ---------------------------------------------------------------------------
# Helpers gráficos
# ---------------------------------------------------------------------------

class _ColorDot(Flowable):
    """Círculo de color sólido para el semáforo en el PDF."""
    def __init__(self, hex_color, diameter=22):
        Flowable.__init__(self)
        self._color = HexColor(hex_color)
        self._d = diameter

    def wrap(self, *args):
        return self._d, self._d

    def draw(self):
        r = self._d / 2
        self.canv.setFillColor(self._color)
        self.canv.circle(r, r, r - 1, fill=1, stroke=0)


def _band_sigma(cached, model_name, pred_vals, test_vals):
    """σ de la banda del 95 %: errores de entrenamiento del modelo; si no existen, errores de prueba."""
    sigma = float(cached.get("extras", {}).get("sigma_train", {}).get(model_name, float("nan")))
    if np.isfinite(sigma) and sigma > 0:
        return sigma, "entrenamiento"
    return float(np.std(np.array(pred_vals) - np.array(test_vals))), "prueba"


def _coverage(cached, model_name):
    """Cobertura empírica de la banda del 95 % sobre el periodo de prueba."""
    pred_vals = [v for v in cached["predictions_dict"].get(model_name, []) if v is not None]
    test_vals = [v for v in cached["test_series"] if v is not None]
    if not pred_vals or len(pred_vals) != len(test_vals):
        return None
    sigma, source = _band_sigma(cached, model_name, pred_vals, test_vals)
    inside = np.abs(np.array(test_vals) - np.array(pred_vals)) <= 1.96 * sigma
    return {"pct": float(inside.mean() * 100), "n_in": int(inside.sum()), "n": int(len(inside)),
            "sigma": sigma, "source": source}


def _forecast_png(cached, best_model, w_cm=17.0, h_cm=7.0):
    """PNG con la serie completa + predicción del mejor modelo + IC 95%."""
    train_s = cached["train_series"]
    test_s  = cached["test_series"]
    pred_s  = cached["predictions_dict"].get(best_model, [])
    dates   = cached["dates"]

    y_train = [x if x is not None else float("nan") for x in train_s]
    y_test  = [x if x is not None else float("nan") for x in test_s]
    y_pred  = [x if x is not None else float("nan") for x in pred_s]
    x_all   = list(range(len(dates)))

    pred_vals = [v for v in pred_s  if v is not None]
    test_vals = [v for v in test_s  if v is not None]
    n_train   = sum(1 for v in train_s if v is not None)

    if pred_vals and len(pred_vals) == len(test_vals):
        sigma = _band_sigma(cached, best_model, pred_vals, test_vals)[0]
        ci_u = [float("nan")] * n_train + [v + 1.96 * sigma for v in pred_vals]
        ci_l = [float("nan")] * n_train + [v - 1.96 * sigma for v in pred_vals]
    else:
        ci_u = ci_l = [float("nan")] * len(dates)

    fig, ax = plt.subplots(figsize=(w_cm / 2.54, h_cm / 2.54))

    ax.plot(x_all, y_train, color="#003087", linewidth=1.2, label="Entrenamiento", zorder=2)
    ax.plot(x_all, y_test,  color="#27AE60", linewidth=1.6, label="Real (test)",   zorder=3)
    ax.plot(x_all, y_pred,  color="#C0392B", linewidth=1.4, linestyle="--",
            label=best_model, zorder=4)
    ax.fill_between(x_all, ci_l, ci_u, color="#C0392B", alpha=0.13, label="IC 95 %", zorder=1)

    n = len(dates)
    step = max(1, n // 8)
    ticks = list(range(0, n, step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([dates[i] for i in ticks], rotation=35, ha="right", fontsize=6)
    ax.set_ylabel("MXN / USD", fontsize=8)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.75)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_title(f"Pronóstico vs. Real — {best_model}  (IC 95 %)", fontsize=9,
                 color="#003087", fontweight="bold", pad=6)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _residuals_png(cached, best_model, w_cm=17.0, h_cm=5.5):
    """PNG con los residuales en el tiempo para el mejor modelo."""
    train_s = cached["train_series"]
    test_s  = cached["test_series"]
    pred_s  = cached["predictions_dict"].get(best_model, [])
    dates   = cached["dates"]

    test_vals = [v for v in test_s if v is not None]
    pred_vals = [v for v in pred_s if v is not None]
    n_train   = sum(1 for v in train_s if v is not None)
    dates_test = dates[n_train:]

    if not pred_vals or len(pred_vals) != len(test_vals):
        return None

    resid  = np.array(pred_vals) - np.array(test_vals)
    colors = ["#C0392B" if r > 0 else "#003087" for r in resid]
    mean_r = float(np.mean(resid))

    fig, ax = plt.subplots(figsize=(w_cm / 2.54, h_cm / 2.54))
    x = list(range(len(resid)))
    ax.bar(x, resid, color=colors, alpha=0.75, width=0.7, zorder=2)
    ax.axhline(0,      color="#C8A800", linewidth=1.2, linestyle="--", zorder=3)
    ax.axhline(mean_r, color="#555555", linewidth=0.8, linestyle=":",
               label=f"Media residual: {mean_r:.4f}", zorder=3)

    step = max(1, len(dates_test) // 6)
    ticks = list(range(0, len(dates_test), step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([dates_test[i] for i in ticks], rotation=35, ha="right", fontsize=6)
    ax.set_ylabel("Residual  (pred − real)", fontsize=8)
    ax.legend(fontsize=7, framealpha=0.75)
    ax.grid(True, alpha=0.2, axis="y", linewidth=0.5)
    ax.set_title(f"Residuales en el tiempo — {best_model}", fontsize=9,
                 color="#003087", fontweight="bold", pad=6)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _semaforo_block(semaforo, avail_w, body_s):
    """Tabla-bloque con el semáforo de riesgo para el PDF."""
    if not semaforo:
        return []

    LEVELS = {
        "low":    ("#2ECC71", "#D5F5E3", "#1A7340"),
        "medium": ("#E67E22", "#FDEBD0", "#7D5A00"),
        "high":   ("#E74C3C", "#FADBD8", "#7B241C"),
    }
    dot_hex, bg_hex, txt_hex = LEVELS.get(semaforo["level"], ("#999999", "#F5F5F5", "#333333"))

    dot = _ColorDot(dot_hex, diameter=22)

    label_s = ParagraphStyle("SL", parent=body_s, fontName="Helvetica-Bold",
                              fontSize=9, textColor=HexColor(txt_hex), leading=13)
    info_s  = ParagraphStyle("SI", parent=body_s, fontSize=8,
                              textColor=HexColor(txt_hex), leading=12)

    content = Table([
        [Paragraph(f"{semaforo['label']}  —  Volatilidad anualizada: {semaforo['annual_vol']} %", label_s)],
        [Spacer(1, 3)],
        [Paragraph(semaforo["desc"], info_s)],
        [Spacer(1, 3)],
        [Paragraph(f"<b>Recomendación:</b> {semaforo['rec']}", info_s)],
    ], colWidths=[avail_w - 42])
    content.setStyle(TableStyle([
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
    ]))

    outer = Table([[dot, content]], colWidths=[32, avail_w - 32])
    outer.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), HexColor(bg_hex)),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
    ]))
    return [outer]


# ---------------------------------------------------------------------------
# Excel
# ---------------------------------------------------------------------------

@bp.route("/download_excel", methods=["POST"])
def download_excel():
    cache_key  = request.form.get("cache_key", "")
    model_name = request.form.get("model_name", "modelo")
    cached = cache.get(cache_key) if cache_key else None

    if not cached:
        return "Sesión expirada o inválida. Vuelve a la página principal y consulta de nuevo.", 400

    train_series = cached["train_series"]
    test_series  = cached["test_series"]
    pred_series  = cached["predictions_dict"].get(model_name, [])
    dates        = cached["dates"]

    df = pd.DataFrame({
        "Date":          dates,
        "Entrenamiento": train_series,
        "Real (test)":   test_series,
        "Pronóstico":    pred_series,
    })

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Datos")
        workbook  = writer.book
        worksheet = writer.sheets["Datos"]
        chart = workbook.add_chart({"type": "line"})
        for idx, col in enumerate(["Entrenamiento", "Real (test)", "Pronóstico"]):
            chart.add_series({
                "name":       col,
                "categories": ["Datos", 1, 0, len(df), 0],
                "values":     ["Datos", 1, idx + 1, len(df), idx + 1],
            })
        worksheet.insert_chart("F2", chart)
    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name=f"{model_name}_resultado.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

@bp.route("/download_pdf", methods=["POST"])
def download_pdf():
    cache_key = request.form.get("cache_key", "")
    cached = cache.get(cache_key) if cache_key else None

    if not cached:
        return "Sesión expirada o inválida. Vuelve a la página principal y consulta de nuevo.", 400

    metrics_df = cached["metrics_df"].copy()
    fc_values  = cached["forecast_values"]
    semaforo   = cached.get("semaforo")

    metrics_df, best_idx = add_ranking(metrics_df)
    best_model = str(metrics_df.loc[best_idx, "Modelo"])

    # ── Colores corporativos ────────────────────────────────────────────────
    UNAM_BLUE  = HexColor("#003087")
    UNAM_GOLD  = HexColor("#C8A800")
    UNAM_LIGHT = HexColor("#F4F6FA")
    WHITE      = rl_colors.white
    DARK       = HexColor("#0D1B3E")
    GRAY       = HexColor("#4A5568")
    GOLD_TINT  = HexColor("#FFFBE6")

    # ── Documento ───────────────────────────────────────────────────────────
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm, topMargin=1.5*cm, bottomMargin=2*cm,
    )
    AVAIL_W = 17 * cm
    base = getSampleStyleSheet()

    def ps(name, **kw):
        return ParagraphStyle(name, parent=base["Normal"], **kw)

    title_s   = ps("T",  fontName="Helvetica-Bold", fontSize=15, textColor=WHITE,     alignment=TA_CENTER, spaceAfter=2)
    sub_s     = ps("Su", fontName="Helvetica",      fontSize=8,  textColor=HexColor("#C8D8FF"), alignment=TA_CENTER, spaceAfter=2)
    section_s = ps("H",  fontName="Helvetica-Bold", fontSize=11, textColor=UNAM_BLUE, spaceBefore=14, spaceAfter=6)
    body_s    = ps("B",  fontName="Helvetica",      fontSize=9,  textColor=DARK,      leading=14)
    small_s   = ps("Sm", fontName="Helvetica",      fontSize=7.5,textColor=GRAY,      leading=11)
    footer_s  = ps("F",  fontName="Helvetica",      fontSize=7,  textColor=GRAY,      alignment=TA_CENTER)

    story = []

    # ── Encabezado ──────────────────────────────────────────────────────────
    hdr = Table([
        [Paragraph("Universidad Nacional Autónoma de México · Facultad de Ingeniería", sub_s)],
        [Paragraph("Reporte de Pronóstico — Tipo de Cambio USD/MXN", title_s)],
        [Paragraph("Modelos de Series de Tiempo, Machine Learning &amp; Deep Learning · Tesis 2026", sub_s)],
    ], colWidths=[AVAIL_W])
    hdr.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), UNAM_BLUE),
        ("TOPPADDING",    (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LEFTPADDING",   (0,0), (-1,-1), 14),
        ("RIGHTPADDING",  (0,0), (-1,-1), 14),
    ]))
    story.append(hdr)
    story.append(HRFlowable(width="100%", thickness=4, color=UNAM_GOLD, spaceAfter=8))

    now_str = datetime.now().strftime("%d/%m/%Y  %H:%M hrs")
    meta = Table([
        ["Fecha de generación:", now_str],
        ["Autor:", "Ing. Ricardo Javier Martínez Suástegui"],
        ["Mejor modelo:", best_model],
    ], colWidths=[4.5*cm, AVAIL_W - 4.5*cm])
    meta.setStyle(TableStyle([
        ("FONT",          (0,0), (0,-1), "Helvetica-Bold", 8),
        ("FONT",          (1,0), (1,-1), "Helvetica",      8),
        ("TEXTCOLOR",     (0,0), (0,-1), UNAM_BLUE),
        ("TEXTCOLOR",     (1,0), (1,-1), DARK),
        ("TOPPADDING",    (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ]))
    story.append(meta)
    story.append(Spacer(1, 0.2*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=UNAM_GOLD, spaceAfter=6))

    # ── Sección 1: Métricas ─────────────────────────────────────────────────
    story.append(Paragraph("1. Métricas Comparativas de Modelos", section_s))
    def _num(v, fmt):
        return fmt.format(v) if v is not None and not pd.isna(v) else "—"

    m_rows = [["Modelo", "MAE", "RMSE", "MAPE", "R²", "DM p", "p aj.", "Veredicto", "Rank"]]
    for _, row in metrics_df.iterrows():
        m_rows.append([
            str(row["Modelo"]),
            _num(row["MAE"], "{:.4f}"), _num(row["RMSE"], "{:.4f}"),
            _num(row["MAPE"], "{:.4f}"), _num(row["R^2"], "{:.4f}"),
            _num(row.get("DM p-valor"), "{:.4f}"), _num(row.get("DM p ajust."), "{:.4f}"),
            str(row.get("Veredicto", "")), str(int(row["Rank"])),
        ])

    col_w = [3.0*cm, 1.8*cm, 1.8*cm, 1.8*cm, 1.8*cm, 1.5*cm, 1.5*cm, 2.7*cm, 1.1*cm]
    m_tbl = Table(m_rows, colWidths=col_w)
    ts = [
        ("BACKGROUND",    (0,0), (-1,0),  UNAM_BLUE),
        ("TEXTCOLOR",     (0,0), (-1,0),  WHITE),
        ("FONT",          (0,0), (-1,0),  "Helvetica-Bold", 8),
        ("FONT",          (0,1), (-1,-1), "Helvetica",      8),
        ("TEXTCOLOR",     (0,1), (-1,-1), DARK),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("ALIGN",         (0,1), (0,-1),  "LEFT"),
        ("GRID",          (0,0), (-1,-1), 0.4, HexColor("#D6DCE8")),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
    ]
    for r in range(1, len(m_rows)):
        if r % 2 == 0:
            ts.append(("BACKGROUND", (0, r), (-1, r), UNAM_LIGHT))
    best_row = list(metrics_df.index).index(best_idx) + 1
    ts += [
        ("BACKGROUND", (0, best_row), (-1, best_row), GOLD_TINT),
        ("TEXTCOLOR",  (0, best_row), (-1, best_row), HexColor("#7A6000")),
        ("FONT",       (0, best_row), (-1, best_row), "Helvetica-Bold", 8),
        ("LINEBEFORE", (0, best_row), (0,  best_row), 4, UNAM_GOLD),
    ]
    m_tbl.setStyle(TableStyle(ts))
    story.append(m_tbl)
    story.append(Spacer(1, 0.15*cm))
    story.append(Paragraph(
        f"Mejor modelo: <b>{best_model}</b> — menor puntuación compuesta en MAE, RMSE, MAPE y R². "
        "Evaluación a un paso adelante; DM p = p-valor de la prueba de Diebold-Mariano frente al modelo ingenuo (Naive); "
        "p aj. = el mismo p-valor con la corrección de Holm por las seis comparaciones simultáneas.",
        small_s,
    ))

    # ── Sección 2: Semáforo de riesgo ───────────────────────────────────────
    story.append(Paragraph("2. Semáforo de Riesgo Cambiario", section_s))
    story.extend(_semaforo_block(semaforo, AVAIL_W, body_s))

    # ── Sección 3: Gráfica del mejor modelo (con IC 95%) ────────────────────
    story.append(Paragraph(f"3. Pronóstico del Mejor Modelo — {best_model}", section_s))
    cov = _coverage(cached, best_model)
    cov_txt = ""
    if cov:
        cov_txt = (f" Cobertura empírica en prueba: {cov['pct']:.1f} % ({cov['n_in']} de {cov['n']} valores reales "
                   f"dentro de la banda; nominal 95 %; sigma = {cov['sigma']:.4f}, errores de {cov['source']}).")
    story.append(Paragraph(
        "Serie histórica completa con la predicción del modelo ganador y su intervalo de confianza al 95 %. "
        "La banda sombreada se obtiene como ±1.96 desviaciones estándar de los errores de entrenamiento a un paso "
        "del modelo, de modo que la cobertura observada en prueba es una verificación honesta." + cov_txt,
        body_s,
    ))
    story.append(Spacer(1, 0.2*cm))
    try:
        fc_buf = _forecast_png(cached, best_model, w_cm=17.0, h_cm=7.0)
        story.append(RLImage(fc_buf, width=AVAIL_W, height=7*cm))
    except Exception:
        story.append(Paragraph("(No fue posible generar la gráfica.)", small_s))

    # ── Sección 4: Residuales en el tiempo ──────────────────────────────────
    story.append(Paragraph(f"4. Análisis de Residuales — {best_model}", section_s))
    story.append(Paragraph(
        "Diferencia entre la predicción y el valor real en cada punto del período de prueba. "
        "Barras rojas indican sobreestimación; barras azules indican subestimación. "
        "La línea dorada marca el cero; la punteada gris indica la media del error.",
        body_s,
    ))
    story.append(Spacer(1, 0.2*cm))
    try:
        res_buf = _residuals_png(cached, best_model, w_cm=17.0, h_cm=5.5)
        if res_buf:
            story.append(RLImage(res_buf, width=AVAIL_W, height=5.5*cm))
        else:
            story.append(Paragraph("(Datos insuficientes para graficar residuales.)", small_s))
    except Exception:
        story.append(Paragraph("(No fue posible generar la gráfica de residuales.)", small_s))

    # ── Sección 5: Valores pronosticados ────────────────────────────────────
    story.append(Paragraph("5. Valores Pronosticados", section_s))
    story.append(Paragraph(
        "Estimación de los próximos puntos del tipo de cambio USD/MXN según cada modelo:", body_s,
    ))
    story.append(Spacer(1, 0.2*cm))

    cell_s     = ps("C",  fontName="Helvetica",      fontSize=9, textColor=DARK,  leading=13, wordWrap="CJK")
    cell_hdr_s = ps("Ch", fontName="Helvetica-Bold", fontSize=8, textColor=WHITE, leading=12)

    fc_rows = [[Paragraph("Modelo", cell_hdr_s), Paragraph("Pronóstico (MXN/USD)", cell_hdr_s)]]
    for model, vals in fc_values.items():
        try:
            val_str = "    ".join(f"{v:.4f}" for v in vals if v is not None)
        except TypeError:
            val_str = f"{vals:.4f}" if vals is not None else "—"
        fc_rows.append([Paragraph(model, cell_s), Paragraph(val_str, cell_s)])

    fc_tbl = Table(fc_rows, colWidths=[4.5*cm, AVAIL_W - 4.5*cm])
    fts = [
        ("BACKGROUND",    (0,0), (-1,0),  UNAM_BLUE),
        ("ALIGN",         (0,0), (-1,-1), "LEFT"),
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
        ("GRID",          (0,0), (-1,-1), 0.4, HexColor("#D6DCE8")),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
    ]
    for r in range(1, len(fc_rows)):
        if r % 2 == 0:
            fts.append(("BACKGROUND", (0, r), (-1, r), UNAM_LIGHT))
    fc_tbl.setStyle(TableStyle(fts))
    story.append(fc_tbl)

    # ── Sección 6: Nota de gestión de riesgo ────────────────────────────────
    story.append(Paragraph("6. Nota de Gestión de Riesgo Cambiario", section_s))
    for item in [
        "Los pronósticos son <b>estimaciones estadísticas</b> basadas en datos históricos, no garantías de comportamiento futuro.",
        "El tipo de cambio USD/MXN está sujeto a factores macroeconómicos, geopolíticos y de mercado que pueden generar desviaciones significativas.",
        "Para decisiones de <b>cobertura cambiaria</b> (forwards, opciones, swaps), se recomienda consultar con un especialista financiero.",
        "Los <b>intervalos de confianza al 95 %</b> del mejor modelo se muestran en la sección 3 de este reporte y en la vista de gráfica de la plataforma web.",
    ]:
        story.append(Paragraph(f"• {item}", body_s))
        story.append(Spacer(1, 0.1*cm))

    # ── Pie de página ────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.4*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=UNAM_GOLD, spaceBefore=4))
    story.append(Paragraph(
        "© 2026  Ing. Ricardo Javier Martínez Suástegui  —  "
        "Universidad Nacional Autónoma de México · Facultad de Ingeniería · Tesis de Licenciatura",
        footer_s,
    ))

    doc.build(story)
    buf.seek(0)

    filename = f"Reporte_USDMXN_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    return send_file(buf, as_attachment=True, download_name=filename, mimetype="application/pdf")
