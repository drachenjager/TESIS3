"""Pipeline de entrenamiento y evaluación de los 7 modelos de pronóstico.

Protocolo uniforme: todos los modelos se estiman solo con el conjunto de
entrenamiento y se evalúan sobre el conjunto de prueba con origen móvil, a un
paso adelante (evaluación principal) y a h = 5 y 20 días (``horizons``). El
modelo ingenuo (paseo aleatorio) es la referencia contra la que se contrasta
cada modelo, en cada horizonte, con la prueba de Diebold-Mariano.
"""

import numpy as np
import pandas as pd

from src.models.ts_models import train_naive, train_sarima, train_holtwinters, _compute_metrics
from src.models.ml_models import train_linear_regression, train_random_forest
from src.models.dl_models import train_rnn, train_lstm
from src.models.evaluation import diebold_mariano
from src.models.horizons import HORIZONS, targets

MODELS = [
    ("Naive",            train_naive),
    ("SARIMA",           train_sarima),
    ("Holt-Winters",     train_holtwinters),
    ("Regresión Lineal", train_linear_regression),
    ("Random Forest",    train_random_forest),
    ("RNN",              train_rnn),
    ("LSTM",             train_lstm),
]
REFERENCE = "Naive"
DM_ALPHA = 0.05
DL_MODELS = {"RNN", "LSTM"}


def _dm_verdict(stat, p):
    if stat is None or p is None or np.isnan(stat) or np.isnan(p):
        return "n/a"
    if p < DM_ALPHA:
        return "mejor que Naive" if stat < 0 else "peor que Naive"
    return "sin diferencia"


def _nan_row(name, h):
    return {"Modelo": name, "h": h, "n_eval": 0, "MAE": float("nan"), "RMSE": float("nan"),
            "MAPE": float("nan"), "R^2": float("nan"), "DM": float("nan"), "DM p-valor": float("nan"),
            "Veredicto": "n/a"}


def train_and_evaluate(df, forecast_steps=1, test_size=5, horizons=HORIZONS, seed=0):
    """Entrena los 7 modelos sobre ``df`` y devuelve métricas, pronósticos y series.

    Returns
    -------
    tuple
        ``(metrics_df, forecast_values, train_series, test_series, predictions_dict, horizons_df)``
        ``metrics_df`` (evaluación a un paso) incluye ``DM``, ``DM p-valor`` y
        ``Veredicto`` frente al modelo ingenuo; ``horizons_df`` repite las métricas
        y el contraste para cada horizonte ``h`` (formato largo: una fila por
        modelo y horizonte, con ``n_eval`` orígenes evaluados).
    """
    ts = df["Close"].values
    if test_size >= len(ts):
        test_size = max(1, len(ts) // 2)

    train_data = ts[:-test_size]
    test_data  = ts[-test_size:]
    horizons = tuple(sorted(set(int(h) for h in horizons) | {1}))

    metrics, preds, fcs, multis = [], {}, {}, {}
    for name, fn in MODELS:
        kwargs = {"horizons": horizons}
        if name in DL_MODELS:
            kwargs["seed"] = seed
        m, p, fc, multi = fn(train_data, test_data, forecast_steps, **kwargs)
        metrics.append(m)
        preds[name] = np.asarray(p, dtype=float)
        fcs[name] = fc
        multis[name] = multi or {}

    ref = preds[REFERENCE]
    for m in metrics:
        name = m["Modelo"]
        if name == REFERENCE:
            stat, p = float("nan"), float("nan")
        else:
            stat, p = diebold_mariano(test_data, preds[name], ref)
        m["DM"] = round(stat, 3) if np.isfinite(stat) else float("nan")
        m["DM p-valor"] = round(p, 4) if np.isfinite(p) else float("nan")
        m["Veredicto"] = "referencia" if name == REFERENCE else _dm_verdict(stat, p)

    metrics_df = pd.DataFrame(metrics)

    # Evaluación por horizonte (h = 1 reproduce metrics_df)
    rows = []
    for h in horizons:
        y_h = targets(test_data, h)
        ref_h = multis[REFERENCE].get(h)
        for name, _ in MODELS:
            p_h = multis[name].get(h)
            if p_h is None or len(y_h) == 0 or len(p_h) != len(y_h) or not np.isfinite(p_h).any():
                rows.append(_nan_row(name, h))
                continue
            mm = _compute_metrics(p_h, y_h, name)
            # La varianza de largo plazo de DM usa h−1 autocovarianzas: con menos de 2h
            # orígenes la prueba no es fiable y se reporta como no disponible.
            if name == REFERENCE or ref_h is None or len(y_h) < max(3, 2 * h):
                stat, pv = float("nan"), float("nan")
            else:
                stat, pv = diebold_mariano(y_h, p_h, ref_h, h=h)
            rows.append({"Modelo": name, "h": h, "n_eval": int(len(y_h)),
                         **{k: mm[k] for k in ("MAE", "RMSE", "MAPE", "R^2")},
                         "DM": round(stat, 3) if np.isfinite(stat) else float("nan"),
                         "DM p-valor": round(pv, 4) if np.isfinite(pv) else float("nan"),
                         "Veredicto": "referencia" if name == REFERENCE else _dm_verdict(stat, pv)})
    horizons_df = pd.DataFrame(rows)

    n_train = len(train_data)
    n_test  = len(test_data)

    def _to_list(arr):
        return [
            float(x) if x is not None and not (isinstance(x, float) and np.isnan(x)) else None
            for x in arr
        ]

    train_series = _to_list(train_data.tolist() + [None] * n_test)
    test_series  = _to_list([None] * n_train + test_data.tolist())
    predictions_dict = {name: [None] * n_train + _to_list(preds[name].tolist()) for name, _ in MODELS}

    return metrics_df, fcs, train_series, test_series, predictions_dict, horizons_df
