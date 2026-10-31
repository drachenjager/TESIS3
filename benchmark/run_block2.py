"""Bloque 2 sobre los 54 experimentos de la tesis (misma serie y particiones del banco):
corrección de Holm, acierto direccional (Pesaran-Timmermann), cobertura empírica del
intervalo del 95 % (sigma de entrenamiento) y prueba de Ljung-Box sobre los residuales.

Usa el pipeline de la plataforma (src/models/pipeline.py), que reproduce el banco
(RNN y LSTM con semilla 0). Guarda results_b2/{P}_{T}.json.

Uso: python benchmark/run_block2.py [P1 P2 ...]
"""
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)
import data                                            # noqa: E402  (benchmark/data.py)
from src.models.pipeline import train_and_evaluate     # noqa: E402
from src.models.evaluation import ljung_box_p          # noqa: E402

OUT = os.path.join(HERE, "results_b2")
os.makedirs(OUT, exist_ok=True)
PERIODS = sys.argv[1:] or ["P1", "P2", "P3", "P4", "P5", "P6"]


def _f(v):
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else float(v)


for P in PERIODS:
    series, dates = data.series_for(P)
    for T in [f"T{i}" for i in range(1, 10)]:
        out = os.path.join(OUT, f"{P}_{T}.json")
        if os.path.exists(out):
            print(f"{P},{T}: ya existe", flush=True)
            continue
        tr, te = data.split(series, T)
        df = pd.DataFrame({"Date": pd.to_datetime(dates), "Close": series})
        t0 = time.time()
        metrics_df, _, _, test_series, preds, extras = train_and_evaluate(df, forecast_steps=1, test_size=len(te))
        y = np.asarray([v for v in test_series if v is not None], dtype=float)
        rec = {"P": P, "T": T, "n": int(len(series)), "n_train": int(len(tr)), "n_test": int(len(te)), "models": {}}
        for _, row in metrics_df.iterrows():
            name = row["Modelo"]
            p = np.asarray([v for v in preds[name] if v is not None], dtype=float)
            sigma = extras["sigma_train"].get(name, float("nan"))
            if len(p) == len(y) and np.isfinite(sigma) and sigma > 0:
                coverage = float(np.mean(np.abs(y - p) <= 1.96 * sigma))
            else:
                coverage = None
            resid = p - y if len(p) == len(y) else np.array([])
            lb_p, lb_lags = ljung_box_p(resid, lags=10) if len(resid) >= 8 else (float("nan"), 0)
            d = extras["direction"].get(name, {})
            rec["models"][name] = {
                "MAE": _f(row["MAE"]), "DM": _f(row["DM"]), "p": _f(row["DM p-valor"]), "p_holm": _f(row["DM p ajust."]),
                "veredicto": row["Veredicto"], "sigma_train": _f(sigma), "coverage": coverage,
                "lb_p": _f(lb_p), "lb_lags": int(lb_lags),
                "dir_hit": _f(d.get("hit")), "dir_n": int(d.get("n", 0)), "pt": _f(d.get("pt")), "pt_p": _f(d.get("p")),
            }
        rec["stationarity"] = {k: {kk: _f(vv) if isinstance(vv, float) else vv for kk, vv in v.items()}
                               for k, v in extras["stationarity"].items() if isinstance(v, dict)}
        rec["seconds"] = round(time.time() - t0, 1)
        json.dump(rec, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        hw = rec["models"]["Holt-Winters"]
        print(f"{P},{T}: {rec['seconds']}s | HW MAE {hw['MAE']} p {hw['p']} holm {hw['p_holm']} cov {hw['coverage']} "
              f"dir {hw['dir_hit']} lb {hw['lb_p']}", flush=True)
print("terminado", flush=True)
