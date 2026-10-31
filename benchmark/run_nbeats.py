"""Agrega N-BEATS (3 semillas) a los 54 experimentos del banco, con las mismas particiones y la
implementación de la plataforma (src/models/dl_models.py). Conserva los resultados anteriores
(incluida la RNN simple, que N-BEATS sustituye) en results/{P}_{T}.json y preds/{P}_{T}.npz.

Uso: python benchmark/run_nbeats.py [P1 P2 ...]
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
import torch

torch.set_num_threads(1)
import data                                      # noqa: E402  (benchmark/data.py)
from src.models.dl_models import train_nbeats    # noqa: E402

SEEDS = [0, 1, 2]
RES = os.path.join(HERE, "results"); PRD = os.path.join(HERE, "preds")
PERIODS = sys.argv[1:] or ["P1", "P2", "P3", "P4", "P5", "P6"]

for P in PERIODS:
    series, dates = data.series_for(P)
    for T in [f"T{i}" for i in range(1, 10)]:
        fj = os.path.join(RES, f"{P}_{T}.json"); fn = os.path.join(PRD, f"{P}_{T}.npz")
        rec = json.load(open(fj, encoding="utf-8"))
        if "N-BEATS" in rec["metrics"] and len(rec["metrics"]["N-BEATS"]) == len(SEEDS):
            print(f"{P},{T}: N-BEATS ya calculado", flush=True)
            continue
        tr, te = data.split(series, T)
        assert len(tr) == rec["n_train"] and len(te) == rec["n_test"]
        preds = dict(np.load(fn))
        t0 = time.time()
        rec["metrics"]["N-BEATS"] = {}
        for sd in SEEDS:
            m, p, _ = train_nbeats(tr, te, 1, seed=sd)
            m.pop("sigma_train", None)
            rec["metrics"]["N-BEATS"][f"seed{sd}"] = m
            preds[f"N-BEATS_seed{sd}"] = np.asarray(p, dtype=float)
        rec.setdefault("tiempos", {})["N-BEATS"] = round(time.time() - t0, 1)
        np.savez(fn, **preds)
        json.dump(rec, open(fj, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        maes = [rec["metrics"]["N-BEATS"][f"seed{s}"]["MAE"] for s in SEEDS]
        print(f"{P},{T}: N-BEATS MAE {maes} | Naive {rec['metrics']['Naive']['MAE']} | LSTM s0 "
              f"{rec['metrics']['LSTM']['seed0']['MAE']} | {rec['tiempos']['N-BEATS']}s", flush=True)
print("terminado", flush=True)
