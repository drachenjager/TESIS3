"""Pipeline de entrenamiento y evaluación de los 7 modelos de pronóstico.

Protocolo uniforme: todos los modelos se estiman solo con el conjunto de
entrenamiento y se evalúan a un paso adelante sobre el conjunto de prueba,
actualizando su estado con el valor real del día anterior. El modelo ingenuo
(paseo aleatorio) es la referencia contra la que se contrasta cada modelo con
la prueba de Diebold-Mariano.
"""

import numpy as np
import pandas as pd

from src.models.ts_models import train_naive, train_sarima, train_holtwinters
from src.models.ml_models import train_linear_regression, train_random_forest
from src.models.dl_models import train_rnn, train_lstm
from src.models.evaluation import diebold_mariano

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


def _dm_verdict(stat, p):
    if stat is None or p is None or np.isnan(stat) or np.isnan(p):
        return "n/a"
    if p < DM_ALPHA:
        return "mejor que Naive" if stat < 0 else "peor que Naive"
    return "sin diferencia"


def train_and_evaluate(df, forecast_steps=1, test_size=5):
    """Entrena los 7 modelos sobre ``df`` y devuelve métricas, pronósticos y series.

    Returns
    -------
    tuple
        ``(metrics_df, forecast_values, train_series, test_series, predictions_dict)``
        ``metrics_df`` incluye las columnas ``DM`` (estadístico de Diebold-Mariano
        frente al modelo ingenuo, negativo = menor error), ``DM p-valor`` y
        ``Veredicto``.
    """
    ts = df["Close"].values
    if test_size >= len(ts):
        test_size = max(1, len(ts) // 2)

    train_data = ts[:-test_size]
    test_data  = ts[-test_size:]

    metrics, preds, fcs = [], {}, {}
    for name, fn in MODELS:
        m, p, fc = fn(train_data, test_data, forecast_steps)
        metrics.append(m)
        preds[name] = np.asarray(p, dtype=float)
        fcs[name] = fc

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

    return metrics_df, fcs, train_series, test_series, predictions_dict
