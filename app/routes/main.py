"""Rutas principales: dashboard, gráfica, residuales, estacionalidad, forecast vs real."""

from datetime import datetime

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from flask import Blueprint, render_template, request

from app import cache
from src.data.fetch import get_usdmxn
from src.features.build import compute_risk
from src.models.evaluation import ljung_box_p
from src.models.pipeline import train_and_evaluate
from src.models.ranking import add_ranking

bp = Blueprint("main", __name__)

VALID_PERIODS = {"3mo", "6mo", "1y", "2y", "5y", "10y", "1mo", "5d", "ytd", "max"}
PERIOD_LABELS = {
    "3mo": "3 meses (corto plazo)",
    "6mo": "6 meses (corto plazo)",
    "1y":  "1 año (mediano plazo)",
    "2y":  "2 años (mediano plazo)",
    "5y":  "5 años (largo plazo)",
    "10y": "10 años (largo plazo)",
    "1mo": "1 mes",
    "5d":  "5 días",
    "ytd": "Año en curso (YTD)",
    "max": "Máximo histórico",
}


def _render_index(**kwargs):
    defaults = dict(
        metrics_table=None, forecast_values=None, predictions_dict=None,
        display_period=None, display_test_percent=None, cache_key=None,
        error=None, selected_period="1y", selected_test_percent=20,
        query_mode="period", selected_start_date="", selected_end_date="",
        selected_anchor_date="", selected_anchor_period="1y",
        semaforo=None, direction_rows=None,
    )
    defaults.update(kwargs)
    return render_template("index.html", **defaults)


def _direction_rows(direction, order):
    """Filas de acierto de dirección (prueba de Pesaran-Timmermann) en el orden del ranking."""
    rows = []
    for name in order:
        d = direction.get(name) or {}
        hit, n, pt, p = d.get("hit"), d.get("n", 0), d.get("pt"), d.get("p")
        if name == "Naive" or hit is None or not np.isfinite(hit):
            rows.append({"Modelo": name, "hit": "—", "n": n, "pt": "—", "p": "—",
                         "verdict": "sin dirección: pronostica 'sin cambio'" if name == "Naive" else "sin dirección definida"})
            continue
        if p is not None and np.isfinite(p):
            verdict = "acierta más que el azar" if p < 0.05 else "no distinguible del azar"
        else:
            verdict = "pronostica siempre la misma dirección; la prueba no aplica"
        rows.append({"Modelo": name, "hit": f"{hit * 100:.1f} %", "n": n,
                     "pt": f"{pt:.2f}" if pt is not None and np.isfinite(pt) else "—",
                     "p": f"{p:.3f}" if p is not None and np.isfinite(p) else "—", "verdict": verdict})
    return rows


def _band_sigma(cached, model_name, pred_values, test_values):
    """σ de la banda del 95 %: errores de entrenamiento del modelo; si no existen, errores de prueba."""
    sigma = float(cached.get("extras", {}).get("sigma_train", {}).get(model_name, float("nan")))
    if np.isfinite(sigma) and sigma > 0:
        return sigma, "entrenamiento"
    return float(np.std(np.array(pred_values) - np.array(test_values))), "prueba"


# ---------------------------------------------------------------------------
# Ruta principal
# ---------------------------------------------------------------------------

@bp.route("/", methods=["GET", "POST"])
def index():
    if request.method != "POST":
        return _render_index()

    query_mode = request.form.get("query_mode", "period")

    try:
        test_percent = float(request.form.get("test_percent", 20))
        if not (1 <= test_percent <= 80):
            raise ValueError()
    except (ValueError, TypeError):
        return _render_index(error="El porcentaje de testing debe estar entre 1 y 80."), 400

    # Estado ancla — solo se usa en modo "anchor"; valores neutros para los otros modos
    anchor_date_str = ""
    anchor_period   = "1y"

    if query_mode == "period":
        selected_period = request.form.get("period_select", "")
        if selected_period not in VALID_PERIODS:
            return _render_index(error="Periodo inválido seleccionado."), 400
        start_date = end_date = None

    elif query_mode == "custom":
        selected_period = None
        start_date = request.form.get("start_date", "").strip()
        end_date   = request.form.get("end_date",   "").strip()
        try:
            dt_start = datetime.strptime(start_date, "%Y-%m-%d")
            dt_end   = datetime.strptime(end_date,   "%Y-%m-%d")
            if dt_start >= dt_end:
                raise ValueError("La fecha de inicio debe ser anterior a la fecha fin.")
            if dt_end > datetime.now():
                raise ValueError("La fecha fin no puede ser futura.")
        except (ValueError, TypeError) as exc:
            return _render_index(
                error=f"Fechas inválidas: {exc}", query_mode="custom",
                selected_start_date=start_date, selected_end_date=end_date,
                selected_test_percent=test_percent,
            ), 400

    elif query_mode == "anchor":
        selected_period = None
        anchor_date_str = request.form.get("anchor_date", "").strip()
        anchor_period   = request.form.get("anchor_period", "").strip()
        if anchor_period not in VALID_PERIODS:
            return _render_index(
                error="Periodo inválido en modo fecha ancla.", query_mode="anchor",
                selected_anchor_date=anchor_date_str, selected_anchor_period=anchor_period,
                selected_test_percent=test_percent,
            ), 400
        try:
            dt_anchor = datetime.strptime(anchor_date_str, "%Y-%m-%d")
            if dt_anchor > datetime.now():
                raise ValueError("La fecha de referencia no puede ser futura.")
            if dt_anchor < datetime(1993, 1, 1):
                raise ValueError("La fecha de referencia mínima disponible es el 01/01/1993.")

            _OFFSETS = {
                "5d":  relativedelta(days=5),
                "1mo": relativedelta(months=1),
                "3mo": relativedelta(months=3),
                "6mo": relativedelta(months=6),
                "1y":  relativedelta(years=1),
                "2y":  relativedelta(years=2),
                "5y":  relativedelta(years=5),
                "10y": relativedelta(years=10),
            }
            if anchor_period == "ytd":
                dt_start = datetime(dt_anchor.year, 1, 1)
            elif anchor_period == "max":
                dt_start = datetime(1993, 1, 1)
            else:
                dt_start = dt_anchor - _OFFSETS[anchor_period]

            if dt_start >= dt_anchor:
                raise ValueError(
                    f"El periodo '{PERIOD_LABELS.get(anchor_period, anchor_period)}' no genera "
                    "un rango válido desde la fecha de referencia indicada."
                )
            start_date = dt_start.strftime("%Y-%m-%d")
            end_date   = anchor_date_str
        except (ValueError, TypeError) as exc:
            return _render_index(
                error=f"Fecha de referencia inválida: {exc}", query_mode="anchor",
                selected_anchor_date=anchor_date_str, selected_anchor_period=anchor_period,
                selected_test_percent=test_percent,
            ), 400

    else:
        return _render_index(error="Modo de consulta inválido."), 400

    today = datetime.now().strftime("%Y%m%d")
    if query_mode == "period":
        cache_key = f"v2_{today}_period_{selected_period}_{int(test_percent)}"
    elif query_mode == "custom":
        cache_key = f"v2_{today}_custom_{start_date}_{end_date}_{int(test_percent)}"
    else:
        cache_key = f"v2_{today}_anchor_{anchor_date_str}_{anchor_period}_{int(test_percent)}"

    cached = cache.get(cache_key)
    if cached is None:
        df = get_usdmxn(period=selected_period, start_date=start_date, end_date=end_date)
        if df is None or len(df) < 10:
            return _render_index(
                error="No se obtuvieron suficientes datos. Prueba con otro periodo o rango de fechas.",
                query_mode=query_mode, selected_period=selected_period,
                selected_start_date=start_date or "", selected_end_date=end_date or "",
                selected_anchor_date=anchor_date_str, selected_anchor_period=anchor_period,
                selected_test_percent=test_percent,
            ), 400

        test_size = max(1, int(len(df) * test_percent / 100))
        metrics_df, forecast_values, train_series, test_series, predictions_dict, extras = (
            train_and_evaluate(df, forecast_steps=test_size, test_size=test_size)
        )
        dates = df["Date"].dt.strftime("%Y-%m-%d").tolist()
        semaforo = compute_risk(df["Close"].values)

        cached = {
            "metrics_df": metrics_df, "forecast_values": forecast_values,
            "train_series": train_series, "test_series": test_series,
            "predictions_dict": predictions_dict, "dates": dates, "semaforo": semaforo,
            "extras": extras,
        }
        cache.set(cache_key, cached)

    # Tabla de métricas con ranking
    metrics_df, best_idx = add_ranking(cached["metrics_df"])

    format_dict = {"MAE": "{:.4f}", "RMSE": "{:.4f}", "MAPE": "{:.4f}", "R^2": "{:.4f}",
                   "DM": "{:.3f}", "DM p-valor": "{:.4f}", "DM p ajust.": "{:.4f}", "Rank": "{:.0f}"}

    def highlight_best(row):
        if row.name == best_idx:
            return [
                "background: linear-gradient(90deg, rgba(200,168,0,0.18), rgba(200,168,0,0.08));"
                "border-left: 4px solid #C8A800; font-weight: 700; color: #7A6000;"
            ] * len(row)
        return [""] * len(row)

    metrics_table = (
        metrics_df.style
        .apply(highlight_best, axis=1)
        .format(format_dict, na_rep="—")
        .hide(axis="index")
        .set_table_styles([
            {"selector": "",           "props": [("border-collapse", "collapse"), ("width", "100%")]},
            {"selector": "thead tr",   "props": [("background", "linear-gradient(90deg,#003087,#005EB8)")]},
            {"selector": "th",         "props": [("color", "#FFFFFF"), ("font-weight", "700"),
                                                  ("font-size", "0.75rem"), ("letter-spacing", "0.08em"),
                                                  ("text-transform", "uppercase"), ("padding", "12px 14px"),
                                                  ("text-align", "center"), ("border", "none")]},
            {"selector": "td",         "props": [("padding", "10px 14px"), ("text-align", "center"),
                                                  ("border-bottom", "1px solid #D6DCE8"),
                                                  ("vertical-align", "middle")]},
            {"selector": "tbody tr:nth-child(even)", "props": [("background-color", "rgba(0,48,135,0.03)")]},
            {"selector": "tbody tr:hover",           "props": [("background-color", "rgba(0,48,135,0.07)")]},
        ])
        .to_html(classes="w-100", table_id="metrics-table")
    )

    def format_forecast(vals):
        try:
            return ", ".join(f"{v:.2f}" for v in vals if v is not None)
        except TypeError:
            return f"{vals:.2f}" if vals is not None else ""

    formatted_forecasts = {m: format_forecast(v) for m, v in cached["forecast_values"].items()}
    if query_mode == "period":
        display_period = PERIOD_LABELS.get(selected_period, selected_period)
    elif query_mode == "anchor":
        display_period = (
            f"{start_date} → {end_date}"
            f"  ({PERIOD_LABELS.get(anchor_period, anchor_period)})"
        )
    else:
        display_period = f"{start_date} → {end_date}"

    return _render_index(
        metrics_table=metrics_table, forecast_values=formatted_forecasts,
        predictions_dict=cached["predictions_dict"], display_period=display_period,
        display_test_percent=test_percent, cache_key=cache_key, query_mode=query_mode,
        selected_period=selected_period, selected_start_date=start_date or "",
        selected_end_date=end_date or "", selected_test_percent=test_percent,
        selected_anchor_date=anchor_date_str, selected_anchor_period=anchor_period,
        semaforo=cached.get("semaforo"),
        direction_rows=_direction_rows(cached.get("extras", {}).get("direction", {}), metrics_df["Modelo"].tolist()),
    )


# ---------------------------------------------------------------------------
# Ruta de gráfica
# ---------------------------------------------------------------------------

@bp.route("/plot", methods=["POST"])
def plot():
    cache_key  = request.form.get("cache_key", "")
    model_name = request.form.get("model_choice", "")
    cached = cache.get(cache_key) if cache_key else None

    if not cached:
        return "Sesión expirada o inválida. Vuelve a la página principal y consulta de nuevo.", 400

    train_series = cached["train_series"]
    test_series  = cached["test_series"]
    pred_series  = cached["predictions_dict"].get(model_name, [])
    dates        = cached["dates"]

    n_train      = sum(1 for x in train_series if x is not None)
    pred_values  = [x for x in pred_series  if x is not None]
    test_values  = [x for x in test_series  if x is not None]

    coverage = None
    if pred_values and test_values and len(pred_values) == len(test_values):
        sigma, sigma_source = _band_sigma(cached, model_name, pred_values, test_values)
        ci_upper = [None] * n_train + [round(v + 1.96 * sigma, 4) for v in pred_values]
        ci_lower = [None] * n_train + [round(v - 1.96 * sigma, 4) for v in pred_values]
        inside = np.abs(np.array(test_values) - np.array(pred_values)) <= 1.96 * sigma
        pct = float(inside.mean() * 100)
        if pct < 90:
            reading = ("La banda es más estrecha de lo que debería: el modelo se equivoca más en prueba que en "
                       "entrenamiento, señal de sobreajuste.")
        elif pct > 98.5:
            reading = "La banda es más ancha de lo necesario: los errores de prueba fueron menores que los de entrenamiento."
        else:
            reading = "La banda es consistente con su nivel nominal del 95 %."
        coverage = {"pct": round(pct, 1), "n_in": int(inside.sum()), "n": int(len(inside)),
                    "sigma": round(sigma, 4), "source": sigma_source, "reading": reading}
    else:
        ci_upper = [None] * len(pred_series)
        ci_lower = [None] * len(pred_series)

    def fmt(series):
        return ", ".join(f"{x:.2f}" for x in series if x is not None)

    return render_template(
        "plot.html",
        model_name=model_name, train_series=train_series, test_series=test_series,
        pred_series=pred_series, ci_upper=ci_upper, ci_lower=ci_lower, dates=dates,
        train_display=fmt(train_series), test_display=fmt(test_series),
        pred_display=fmt(pred_series), cache_key=cache_key, coverage=coverage,
    )


# ---------------------------------------------------------------------------
# Análisis de residuales
# ---------------------------------------------------------------------------

@bp.route("/residuals", methods=["POST"])
def residuals():
    cache_key    = request.form.get("cache_key", "")
    model_filter = request.form.get("model_choice", "").strip()
    cached = cache.get(cache_key) if cache_key else None

    if not cached:
        return "Sesión expirada o inválida. Vuelve a la página principal y consulta de nuevo.", 400

    train_series     = cached["train_series"]
    test_series      = cached["test_series"]
    predictions_dict = cached["predictions_dict"]
    dates            = cached["dates"]

    n_train     = sum(1 for x in train_series if x is not None)
    test_values = [x for x in test_series if x is not None]
    dates_test  = dates[n_train:]

    residuals_data = {}
    residuals_stats = {}
    hist_data = {}
    interpretations = {}

    for model_name, pred_series in predictions_dict.items():
        if model_filter and model_name != model_filter:
            continue
        pred_values = [x for x in pred_series if x is not None]
        if not pred_values or len(pred_values) != len(test_values):
            continue

        resid = np.array(pred_values, dtype=float) - np.array(test_values, dtype=float)
        residuals_data[model_name] = [round(float(r), 4) for r in resid]

        mean_r = float(np.mean(resid))
        std_r  = float(np.std(resid))
        mae_r  = float(np.mean(np.abs(resid)))
        residuals_stats[model_name] = {
            "mean": round(mean_r, 4), "std": round(std_r, 4), "mae": round(mae_r, 4),
            "max": round(float(np.max(resid)), 4), "min": round(float(np.min(resid)), 4),
        }

        counts, edges = np.histogram(resid, bins=10)
        hist_data[model_name] = {
            "counts": counts.tolist(),
            "labels": [f"{float(edges[i]):.3f}" for i in range(len(edges) - 1)],
        }

        if abs(mean_r) < 0.05:
            bias_txt = "La media de residuales es cercana a cero, por lo que el modelo es prácticamente insesgado."
        elif mean_r > 0:
            bias_txt = f"La media positiva ({mean_r:.4f}) indica que el modelo tiende a <strong>sobreestimar</strong> el tipo de cambio."
        else:
            bias_txt = f"La media negativa ({mean_r:.4f}) indica que el modelo tiende a <strong>subestimar</strong> el tipo de cambio."

        cv = std_r / mae_r if mae_r > 0 else 0
        if cv < 0.6:
            var_txt = "La baja variabilidad de los errores refleja predicciones consistentes."
        elif cv < 1.1:
            var_txt = "Se observa variabilidad moderada en los errores."
        else:
            var_txt = "La alta variabilidad sugiere que el modelo presenta dificultades para capturar ciertos patrones de la serie."

        lb_p, lb_lags = ljung_box_p(resid, lags=10)
        residuals_stats[model_name]["lb_p"] = f"{lb_p:.3f}" if np.isfinite(lb_p) else "—"
        residuals_stats[model_name]["lb_lags"] = lb_lags
        if not np.isfinite(lb_p):
            lb_txt = ""
        elif lb_p >= 0.05:
            lb_txt = (f" Prueba de Ljung-Box (p = {lb_p:.3f}, {lb_lags} rezagos): los residuales <strong>no muestran "
                      "autocorrelación</strong>; lo que queda es ruido sin estructura que otro modelo pudiera aprovechar.")
        else:
            lb_txt = (f" Prueba de Ljung-Box (p = {lb_p:.3f}, {lb_lags} rezagos): los residuales <strong>conservan "
                      "autocorrelación</strong>; quedan patrones que el modelo no capturó.")
        interpretations[model_name] = f"{bias_txt} {var_txt}{lb_txt}"

    return render_template(
        "residuals.html",
        residuals_data=residuals_data, residuals_stats=residuals_stats,
        hist_data=hist_data, interpretations=interpretations,
        dates_test=dates_test, cache_key=cache_key,
        stationarity=cached.get("extras", {}).get("stationarity"),
    )


# ---------------------------------------------------------------------------
# Historial de pronósticos vs. realidad
# ---------------------------------------------------------------------------

@bp.route("/forecast_vs_real", methods=["POST"])
def forecast_vs_real():
    cache_key = request.form.get("cache_key", "")
    cached = cache.get(cache_key) if cache_key else None

    if not cached:
        return "Sesión expirada o inválida. Vuelve a la página principal y consulta de nuevo.", 400

    train_series     = cached["train_series"]
    test_series      = cached["test_series"]
    predictions_dict = cached["predictions_dict"]
    dates            = cached["dates"]

    train_values = [float(x) for x in train_series if x is not None]
    n_train      = len(train_values)
    test_values  = [float(x) for x in test_series  if x is not None]
    n_test       = len(test_values)
    dates_test   = dates[len(dates) - n_test:]
    dates_all    = dates

    model_preds = {}
    for model, pred_series in predictions_dict.items():
        vals = [float(x) for x in pred_series if x is not None]
        if len(vals) == n_test:
            model_preds[model] = vals

    accuracy_stats = {}
    for model, preds in model_preds.items():
        errs_pct = [abs(p - r) / max(abs(r), 1e-10) * 100 for p, r in zip(preds, test_values)]
        mae = float(np.mean(np.abs(np.array(preds) - np.array(test_values))))
        accuracy_stats[model] = {
            "mae":         round(mae, 4),
            "within_1pct": round(sum(1 for e in errs_pct if e <= 1) / len(errs_pct) * 100, 1),
            "within_2pct": round(sum(1 for e in errs_pct if e <= 2) / len(errs_pct) * 100, 1),
            "within_5pct": round(sum(1 for e in errs_pct if e <= 5) / len(errs_pct) * 100, 1),
        }

    best_model = min(accuracy_stats, key=lambda m: accuracy_stats[m]["mae"]) if accuracy_stats else ""

    err_pct_matrix = {
        model: [round(abs(p - r) / max(abs(r), 1e-10) * 100, 3) for p, r in zip(preds, test_values)]
        for model, preds in model_preds.items()
    }

    return render_template(
        "forecast_vs_real.html",
        dates_all=dates_all, dates_test=dates_test, n_train=n_train,
        train_values=[round(v, 4) for v in train_values],
        test_values=[round(v, 4) for v in test_values],
        model_preds={m: [round(v, 4) for v in p] for m, p in model_preds.items()},
        err_pct_matrix=err_pct_matrix, accuracy_stats=accuracy_stats,
        best_model=best_model, model_names=list(model_preds.keys()), cache_key=cache_key,
    )


# ---------------------------------------------------------------------------
# Análisis de estacionalidad
# ---------------------------------------------------------------------------

@bp.route("/seasonality", methods=["POST"])
def seasonality():
    cache_key = request.form.get("cache_key", "")
    cached = cache.get(cache_key) if cache_key else None

    if not cached:
        return "Sesión expirada o inválida. Vuelve a la página principal y consulta de nuevo.", 400

    dates        = cached["dates"]
    train_series = cached["train_series"]
    test_series  = cached["test_series"]

    full_prices = [float(t if t is not None else v) for t, v in zip(train_series, test_series)]
    df = pd.DataFrame({"Date": pd.to_datetime(dates), "Close": full_prices})
    df["Month"]   = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    df["Year"]    = df["Date"].dt.year

    MONTH_NAMES   = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
    QUARTER_NAMES = ["Q1 (Ene–Mar)","Q2 (Abr–Jun)","Q3 (Jul–Sep)","Q4 (Oct–Dic)"]

    mon = df.groupby("Month")["Close"].agg(mean="mean", std="std", min_val="min", max_val="max").reset_index()
    mon["std"]   = mon["std"].fillna(0)
    mon["label"] = mon["Month"].apply(lambda m: MONTH_NAMES[m - 1])

    qtr = df.groupby("Quarter")["Close"].agg(mean="mean", std="std", min_val="min", max_val="max").reset_index()
    qtr["std"]   = qtr["std"].fillna(0)
    qtr["label"] = qtr["Quarter"].apply(lambda q: QUARTER_NAMES[q - 1])

    yr = df.groupby("Year")["Close"].agg(mean="mean", std="std").reset_index()
    yr["std"] = yr["std"].fillna(0)

    insights = {
        "most_volatile_month": mon.loc[int(mon["std"].idxmax()),  "label"],
        "most_volatile_std":   round(float(mon["std"].max()),     4),
        "most_stable_month":   mon.loc[int(mon["std"].idxmin()),  "label"],
        "most_stable_std":     round(float(mon["std"].min()),     4),
        "highest_avg_month":   mon.loc[int(mon["mean"].idxmax()), "label"],
        "highest_avg_val":     round(float(mon["mean"].max()),    4),
        "lowest_avg_month":    mon.loc[int(mon["mean"].idxmin()), "label"],
        "lowest_avg_val":      round(float(mon["mean"].min()),    4),
        "total_years":         int(yr["Year"].nunique()),
        "show_yearly":         int(yr["Year"].nunique()) >= 2,
    }

    return render_template(
        "seasonality.html",
        monthly_labels=mon["label"].tolist(), monthly_mean=mon["mean"].round(4).tolist(),
        monthly_std=mon["std"].round(4).tolist(), monthly_min=mon["min_val"].round(4).tolist(),
        monthly_max=mon["max_val"].round(4).tolist(), quarterly_labels=qtr["label"].tolist(),
        quarterly_mean=qtr["mean"].round(4).tolist(), quarterly_std=qtr["std"].round(4).tolist(),
        yearly_labels=yr["Year"].astype(str).tolist(), yearly_mean=yr["mean"].round(4).tolist(),
        yearly_std=yr["std"].round(4).tolist(), insights=insights, cache_key=cache_key,
    )
