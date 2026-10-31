"""Banco de Tesis: los 54 experimentos publicados en el Capítulo 4 y el Apéndice C.

Página de solo lectura. Los resultados provienen de ``src/benchmark_data/resultados_tesis.json``
(generado por ``scripts/build_published.py`` a partir de ``benchmark/``); aquí no se entrena nada.
"""

import json
import os

from flask import Blueprint, render_template

bp = Blueprint("banco", __name__)

_PUB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "src", "benchmark_data", "resultados_tesis.json")
_PUB = None
_CTX = None

MODELS6 = ["SARIMA", "Holt-Winters", "Regresión Lineal", "Random Forest", "RNN", "LSTM"]
ABBR = {"Naive": "Naive", "SARIMA": "SARIMA", "Holt-Winters": "H-W", "Regresión Lineal": "Reg. Lin.",
        "Random Forest": "R. Forest", "RNN": "RNN", "LSTM": "LSTM"}
FAMILY = {"Naive": "Referencia", "SARIMA": "Series de tiempo", "Holt-Winters": "Series de tiempo",
          "Regresión Lineal": "Machine Learning", "Random Forest": "Machine Learning",
          "RNN": "Deep Learning", "LSTM": "Deep Learning"}
EQUAL_TOL = 0.05        # % de diferencia de MAE frente a Naive por debajo del cual se considera "igual"


def published():
    global _PUB
    if _PUB is None:
        with open(_PUB_PATH, encoding="utf-8") as fh:
            _PUB = json.load(fh)
    return _PUB


def _rel(mae, naive_mae):
    """Diferencia porcentual del MAE frente al modelo ingenuo (negativo = mejor que Naive)."""
    if mae is None or not naive_mae:
        return None
    return (mae / naive_mae - 1.0) * 100.0


def _cell_class(rel, winner_verdict):
    if winner_verdict == "mejor que Naive":
        return "sig"
    if rel is None:
        return "na"
    if rel < -EQUAL_TOL:
        return "better"
    if rel <= EQUAL_TOL:
        return "equal"
    return "worse"


def _lectura(e, models, naive):
    """Una frase de lectura para el elemento, construida solo con hechos de la tabla."""
    top7 = [m["Modelo"] for m in models if m["rank7"] == 1]
    best6 = [m for m in models if m.get("rank6") == 1]
    parts = []
    if "Naive" in top7:
        others = [ABBR[m] for m in top7 if m != "Naive"]
        parts.append("El modelo ingenuo queda en primer lugar" + (f" (empatado con {', '.join(others)})" if others else "") + ".")
    else:
        parts.append(f"Primer lugar en el ranking con los 7 modelos: {', '.join(ABBR[m] for m in top7)}.")
    if best6:
        b = best6[0]
        rel = _rel(b["MAE"], naive["MAE"])
        rel_txt = ("igual que Naive" if rel is not None and abs(rel) <= EQUAL_TOL
                   else f"{rel:+.1f} % de MAE frente a Naive" if rel is not None else "")
        parts.append(f"Mejor de los seis propuestos: {', '.join(ABBR[m['Modelo']] for m in best6)} (MAE {b['MAE']:.4f}, {rel_txt}).")
    sig_better = [ABBR[m["Modelo"]] for m in models if m["veredicto"] == "mejor que Naive"]
    sig_worse = [ABBR[m["Modelo"]] for m in models if m["veredicto"] == "peor que Naive"]
    if sig_better:
        parts.append(f"Supera a Naive con significancia: {', '.join(sig_better)}.")
        for m in models:
            if m["veredicto"] != "mejor que Naive":
                continue
            ph, pb = m.get("p_holm"), m.get("p_holm_bank")
            if ph is not None and ph >= 0.05:
                parts.append(f"Con la corrección de Holm por las seis comparaciones del experimento "
                             f"(p ajustado = {ph:.3f}) esa ventaja deja de ser significativa.")
            elif ph is not None:
                parts.append(f"La ventaja se mantiene con la corrección de Holm dentro del experimento (p ajustado = {ph:.3f}, "
                             f"porque las otras comparaciones son extremas)"
                             + (f", pero no al considerar las pruebas del banco completo (p ajustado = {pb:.2f})." if pb is not None and pb >= 0.05 else "."))
    else:
        parts.append("Ningún modelo supera a Naive con significancia estadística.")
    if sig_worse:
        parts.append(f"Significativamente peores que Naive: {', '.join(sig_worse)}.")
    sarima = next((m for m in models if m["Modelo"] == "SARIMA"), None)
    if sarima and sarima["MAE"] == naive["MAE"] and "(0, 1, 0)" in e.get("sarima_order", ""):
        parts.append("auto_arima eligió (0,1,0): SARIMA es idéntico al paseo aleatorio.")
    return " ".join(parts)


def _build_context(pub):
    periods, tests = [], [{"T": T, "pct": pct} for T, pct in pub["tests"].items()]
    for P, per in pub["periods"].items():
        e1 = pub["elements"][f"{P},T1"]
        periods.append({"P": P, "key": per["key"], "label": per["label"], "n": e1["n"],
                        "fecha_inicio": e1["fecha_inicio"], "fecha_fin": e1["fecha_fin"]})

    heat, rank1, details = [], [], []
    dm_counts = {m: {"mejor que Naive": 0, "sin diferencia": 0, "peor que Naive": 0, "n/a": 0, "mejor (Holm)": 0}
                 for m in MODELS6}
    for per in periods:
        P = per["P"]
        cells, counts, elems = [], {m: 0 for m in MODELS6}, []
        for t in tests:
            el = f"{P},{t['T']}"
            e = pub["elements"][el]
            models = sorted(e["models"], key=lambda m: (m["rank7"], m["MAE"] if m["MAE"] is not None else 9e9))
            by = {m["Modelo"]: m for m in models}
            naive = by["Naive"]
            winners = [m for m in models if m.get("rank6") == 1]
            w = winners[0] if winners else None
            rel = _rel(w["MAE"], naive["MAE"]) if w else None
            cls = _cell_class(rel, w["veredicto"] if w else None)
            for m in models:
                if m["Modelo"] != "Naive":
                    dm_counts[m["Modelo"]][m["veredicto"]] += 1
                    if m["veredicto"] == "mejor que Naive" and m.get("p_holm") is not None and m["p_holm"] < 0.05:
                        dm_counts[m["Modelo"]]["mejor (Holm)"] += 1
                if m.get("rank6") == 1:
                    counts[m["Modelo"]] += 1
            cells.append({
                "el": el, "anchor": el.replace(",", "-"), "T": t["T"], "pct": t["pct"],
                "winners": [ABBR[m["Modelo"]] for m in winners],
                "mae": w["MAE"] if w else None, "naive_mae": naive["MAE"], "rel": rel, "cls": cls,
                "naive_first": naive["rank7"] == 1,
                "title": (f"{el}: historia de {per['label']}, {t['pct']} % de prueba (n = {e['n']}). "
                          f"Mejor de los seis: {', '.join(ABBR[m['Modelo']] for m in winners)} con MAE {w['MAE']:.4f}; "
                          f"Naive {naive['MAE']:.4f}." if w else el),
            })
            max_mae = max(m["MAE"] for m in models if m["MAE"] is not None)
            rows = []
            for m in models:
                r = _rel(m["MAE"], naive["MAE"])
                rows.append({**m, "abbr": ABBR[m["Modelo"]], "family": FAMILY[m["Modelo"]],
                             "bar": (m["MAE"] / max_mae * 100.0) if m["MAE"] is not None and max_mae else 0.0,
                             "rel": r, "is_naive": m["Modelo"] == "Naive"})
            elems.append({"el": el, "anchor": el.replace(",", "-"), "T": t["T"], "pct": t["pct"], "n": e["n"],
                          "n_train": e["n_train"], "n_test": e["n_test"], "fecha_inicio": e["fecha_inicio"],
                          "fecha_test": e["fecha_test"], "fecha_fin": e["fecha_fin"],
                          "sarima_order": e.get("sarima_order", ""), "rows": rows,
                          "lectura": _lectura(e, models, naive)})
        heat.append({**per, "cells": cells})
        rank1.append({**per, "counts": counts, "naive_first": sum(1 for c in cells if c["naive_first"])})
        details.append({**per, "elements": elems})

    rank1_total = {m: sum(r["counts"][m] for r in rank1) for m in MODELS6}
    span = {"inicio": min(p["fecha_inicio"] for p in periods), "fin": max(p["fecha_fin"] for p in periods)}
    summary = {
        "n_elements": pub["summary"]["n_elements"],
        "naive_first": sum(r["naive_first"] for r in rank1),
        "better": pub["summary"]["better_than_naive"],
        "better_holm": pub["summary"].get("better_holm", 0),
        "better_holm_bank": pub["summary"].get("better_holm_bank", 0),
        "n_tests": pub["summary"].get("n_tests", 0),
        "worse": pub["summary"]["worse_than_naive"],
    }
    return dict(pub=pub, periods=periods, tests=tests, heat=heat, rank1=rank1, rank1_total=rank1_total,
                dm_counts=dm_counts, details=details, span=span, summary=summary,
                models6=MODELS6, abbr=ABBR, anchor=pub["anchor"])


@bp.route("/banco")
def banco():
    global _CTX
    if _CTX is None:
        _CTX = _build_context(published())
    return render_template("banco.html", **_CTX)
