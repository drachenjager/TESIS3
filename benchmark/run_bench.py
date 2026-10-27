"""Banco de pruebas: 9 experimentos (T1..T9) para el periodo P indicado.

Uso: python run_bench.py P3
Guarda results/{P}_{T}.json (métricas) y preds/{P}_{T}.npz (predicciones).
"""
import sys, os, json, time, warnings, traceback
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch
torch.set_num_threads(1)
import data
from models import ts_models, ml_models, dl_models

P = sys.argv[1]
SEEDS = [0, 1, 2]
RES = os.path.join(HERE, "results"); PRD = os.path.join(HERE, "preds")
os.makedirs(RES, exist_ok=True); os.makedirs(PRD, exist_ok=True)
log = open(os.path.join(HERE, f"log_{P}.txt"), "a", encoding="utf-8")


def say(msg):
    print(msg, flush=True); log.write(msg + "\n"); log.flush()


def sarima_both(tr, te):
    """auto_arima una sola vez; devuelve (one-step, multistep, orden)."""
    from pmdarima import auto_arima
    model = auto_arima(tr, max_p=3, max_q=3, max_P=1, max_Q=1, seasonal=len(tr) >= 20, m=5,
                       stepwise=True, suppress_warnings=True, error_action="ignore",
                       information_criterion="aic")
    one, _ = ts_models._sarima_one_step(model.arima_res_, tr, te, 1)
    multi = np.asarray(model.predict(n_periods=len(te)), dtype=float)
    return one, multi, f"{model.order}x{model.seasonal_order}"


series, dates = data.series_for(P)
for T in [f"T{i}" for i in range(1, 10)]:
    out = os.path.join(RES, f"{P}_{T}.json")
    if os.path.exists(out):
        say(f"{P},{T}: ya existe, se omite"); continue
    tr, te = data.split(series, T)
    rec = {"P": P, "T": T, "n": int(len(series)), "n_train": int(len(tr)), "n_test": int(len(te)),
           "fecha_inicio": str(dates[0])[:10], "fecha_test": str(dates[len(tr)])[:10],
           "fecha_fin": str(dates[-1])[:10], "metrics": {}, "tiempos": {}}
    preds = {"y_test": te}
    t0 = time.time()

    m, p, _ = ts_models.train_naive(tr, te); rec["metrics"]["Naive"] = m; preds["Naive"] = p

    t1 = time.time()
    try:
        one, multi, order = sarima_both(tr, te)
        rec["metrics"]["SARIMA"] = ts_models._compute_metrics(one, te, "SARIMA")
        rec["metrics"]["SARIMA_multistep"] = ts_models._compute_metrics(multi, te, "SARIMA")
        rec["sarima_order"] = order
        preds["SARIMA"] = one; preds["SARIMA_multistep"] = multi
    except Exception:
        say(f"{P},{T}: SARIMA falló\n{traceback.format_exc()}")
        rec["metrics"]["SARIMA"] = ts_models._empty_metrics("SARIMA", len(te), 1)[0]
        rec["metrics"]["SARIMA_multistep"] = ts_models._empty_metrics("SARIMA", len(te), 1)[0]
    rec["tiempos"]["SARIMA"] = round(time.time() - t1, 1)

    m, p, _ = ts_models.train_holtwinters(tr, te); rec["metrics"]["Holt-Winters"] = m; preds["Holt-Winters"] = p
    m, p, _ = ts_models.train_holtwinters_multistep(tr, te); rec["metrics"]["Holt-Winters_multistep"] = m; preds["Holt-Winters_multistep"] = p
    m, p, _ = ml_models.train_linear_regression(tr, te); rec["metrics"]["Regresión Lineal"] = m; preds["Regresión Lineal"] = p
    m, p, _ = ml_models.train_random_forest(tr, te); rec["metrics"]["Random Forest"] = m; preds["Random Forest"] = p

    t1 = time.time()
    for name, fn in [("RNN", dl_models.train_rnn), ("LSTM", dl_models.train_lstm)]:
        rec["metrics"][name] = {}
        for sd in SEEDS:
            m, p, _ = fn(tr, te, seed=sd)
            rec["metrics"][name][f"seed{sd}"] = m; preds[f"{name}_seed{sd}"] = p
    rec["tiempos"]["DL"] = round(time.time() - t1, 1)
    rec["tiempos"]["total"] = round(time.time() - t0, 1)

    np.savez(os.path.join(PRD, f"{P}_{T}.npz"), **{k: np.asarray(v, dtype=float) for k, v in preds.items()})
    json.dump(rec, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    lr = rec["metrics"]["Regresión Lineal"]["MAE"]; nv = rec["metrics"]["Naive"]["MAE"]
    sa = rec["metrics"]["SARIMA"]["MAE"]; sm = rec["metrics"]["SARIMA_multistep"]["MAE"]
    say(f"{P},{T}: n={len(series)} train={len(tr)} test={len(te)} | MAE LR={lr} Naive={nv} "
        f"SARIMA 1paso={sa} (multi={sm}) [{rec.get('sarima_order','?')}] | {rec['tiempos']['total']}s")
say(f"{P}: terminado")
