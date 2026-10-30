"""Banco de pruebas de la tesis: los 54 experimentos publicados (Capítulo 4 y
Apéndice C) y su re-ejecución en vivo para contrastar publicado vs. reproducido."""

import json
import os
from datetime import datetime

from flask import Blueprint, render_template, request

from app.routes.main import (CACHE_PREFIX, DATA_SOURCES, _render_index, period_bounds,
                             results_context, run_experiment)
from src.data.frozen import FROZEN_ANCHOR
from src.models.ranking import add_ranking

bp = Blueprint("banco", __name__)

_PUB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "src", "benchmark_data", "resultados_tesis.json")
_PUB = None
MATCH_TOL = 0.0005        # |ΔMAE| para considerar que una corrida reproduce el valor publicado
NEAR_TOL = 0.002          # ruido de punto flotante (p. ej. umbrales de Random Forest)


def published():
    global _PUB
    if _PUB is None:
        with open(_PUB_PATH, encoding="utf-8") as fh:
            _PUB = json.load(fh)
    return _PUB


def _matrix(pub):
    rows = []
    for P, per in pub["periods"].items():
        cells = []
        for T, pct in pub["tests"].items():
            el = f"{P},{T}"
            e = pub["elements"][el]
            models = {m["Modelo"]: m for m in e["models"]}
            winners = [m["Modelo"] for m in e["models"] if m.get("rank6") == 1]
            naive = models["Naive"]
            cells.append({
                "el": el, "anchor": f"{P}-{T}", "pct": pct, "winners": winners,
                "mae": min(models[w]["MAE"] for w in winners) if winners else None,
                "naive_mae": naive["MAE"], "naive_best": naive["rank7"] == 1,
                "significant": any(m["veredicto"] == "mejor que Naive" for m in e["models"]),
            })
        rows.append({"P": P, "label": per["label"], "cells": cells})
    return rows


@bp.route("/banco")
def banco():
    pub = published()
    worse = sorted(pub["summary"]["worse_than_naive"].items(), key=lambda kv: -kv[1])
    return render_template(
        "banco.html", pub=pub, matrix=_matrix(pub), worse=worse,
        data_sources=DATA_SOURCES, anchor=pub["anchor"],
    )


@bp.route("/banco/run", methods=["POST"])
def banco_run():
    pub = published()
    el = request.form.get("element", "").strip()
    data_source = request.form.get("data_source", "tesis")
    e = pub["elements"].get(el)
    if e is None or data_source not in DATA_SOURCES:
        return "Elemento o fuente de datos inválidos.", 400

    start, end = period_bounds(FROZEN_ANCHOR, e["period_key"])
    start_date, end_date = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    today = datetime.now().strftime("%Y%m%d")
    cache_key = f"{CACHE_PREFIX}_{today}_banco_{e['P']}{e['T']}_{data_source}"

    cached, err = run_experiment(cache_key, data_source, None, start_date, end_date, e["test_pct"])
    if err:
        return _render_index(error=err, data_source=data_source), 400

    rep, _ = add_ranking(cached["metrics_df"])
    rep = rep.set_index("Modelo")
    rows, n_match = [], 0
    for m in e["models"]:
        name = m["Modelo"]
        r = rep.loc[name]
        seed0 = m.get("seed0")
        ref_mae = seed0["MAE"] if seed0 else m["MAE"]
        ref_r2 = seed0["R^2"] if seed0 else m["R^2"]
        rep_mae = None if r["MAE"] != r["MAE"] else float(r["MAE"])
        delta = None if (rep_mae is None or ref_mae is None) else rep_mae - ref_mae
        if delta is None:
            status = "n/a"
        elif abs(delta) <= MATCH_TOL:
            status = "coincide"; n_match += 1
        elif abs(delta) <= NEAR_TOL:
            status = "≈ coincide"; n_match += 1
        else:
            status = "difiere"
        rows.append({
            "Modelo": name,
            "pub_mae": m["MAE"], "pub_r2": m["R^2"], "pub_p": m["p"], "pub_rank": m["rank7"],
            "pub_seed0": bool(seed0), "ref_mae": ref_mae, "ref_r2": ref_r2,
            "rep_mae": rep_mae, "rep_r2": None if r["R^2"] != r["R^2"] else float(r["R^2"]),
            "rep_p": None if r["DM p-valor"] != r["DM p-valor"] else float(r["DM p-valor"]),
            "rep_rank": int(r["Rank"]), "delta": delta, "status": status,
        })

    banco_compare = {
        "element": el, "info": e, "rows": rows, "n_match": n_match, "n_total": len(rows),
        "data_source_label": DATA_SOURCES[data_source], "anchor": pub["anchor"],
    }
    display_period = f"{start_date} → {end_date}  (historia de {e['periodo']}, elemento {el} de la tesis)"
    return _render_index(
        **results_context(cached),
        banco_compare=banco_compare, display_period=display_period,
        display_test_percent=e["test_pct"], cache_key=cache_key,
        query_mode="anchor", selected_anchor_date=pub["anchor"], selected_anchor_period=e["period_key"],
        selected_test_percent=e["test_pct"], data_source=data_source,
    )
