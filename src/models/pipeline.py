"""Pipeline de entrenamiento y evaluación de los 7 modelos de pronóstico.

Protocolo uniforme: todos los modelos se estiman solo con el conjunto de
entrenamiento y se evalúan a un paso adelante sobre el conjunto de prueba,
actualizando su estado con el valor real del día anterior. El modelo ingenuo
(paseo aleatorio) es la referencia contra la que se contrasta cada modelo con
la prueba de Diebold-Mariano; los p-valores se reportan sin ajustar y con la
corrección de Holm por las comparaciones simultáneas del experimento.
"""

import numpy as np
import pandas as pd

from src.models.ts_models import train_naive, train_sarima, train_holtwinters
from src.models.ml_models import train_linear_regression, train_random_forest
from src.models.dl_models import train_rnn, train_lstm
from src.models.evaluation import diebold_mariano, holm_adjust, pesaran_timmermann, stationarity_tests

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
        ``(metrics_df, forecast_values, train_series, test_series, predictions_dict, extras)``
        ``metrics_df`` incluye ``DM`` (estadístico de Diebold-Mariano frente al
        modelo ingenuo, negativo = menor error), ``DM p-valor``, ``DM p ajust.``
        (corrección de Holm) y ``Veredicto``. ``extras`` reúne diagnósticos:
        ``sigma_train`` (desviación de los errores de entrenamiento por modelo,
        para el intervalo de confianza), ``direction`` (acierto de dirección y
        prueba de Pesaran-Timmermann) y ``stationarity`` (ADF/KPSS de la serie).
    """
    ts = df["Close"].values
    if test_size >= len(ts):
        test_size = max(1, len(ts) // 2)

    train_data = ts[:-test_size]
    test_data  = ts[-test_size:]

    metrics, preds, fcs, sigma_train = [], {}, {}, {}
    for name, fn in MODELS:
        m, p, fc = fn(train_data, test_data, forecast_steps)
        sigma_train[name] = float(m.pop("sigma_train", float("nan")))
        metrics.append(m)
        preds[name] = np.asarray(p, dtype=float)
        fcs[name] = fc

    ref = preds[REFERENCE]
    raw_p = []
    for m in metrics:
        name = m["Modelo"]
        if name == REFERENCE:
            stat, p = float("nan"), float("nan")
        else:
            stat, p = diebold_mariano(test_data, preds[name], ref)
        m["DM"] = round(stat, 3) if np.isfinite(stat) else float("nan")
        m["DM p-valor"] = round(p, 4) if np.isfinite(p) else float("nan")
        m["Veredicto"] = "referencia" if name == REFERENCE else _dm_verdict(stat, p)
        raw_p.append(p)
    for m, p_adj in zip(metrics, holm_adjust(raw_p)):
        m["DM p ajust."] = round(p_adj, 4) if np.isfinite(p_adj) else float("nan")
    cols = ["Modelo", "MAE", "RMSE", "MAPE", "R^2", "DM", "DM p-valor", "DM p ajust.", "Veredicto"]
    metrics_df = pd.DataFrame(metrics)[cols]

    # Acierto de dirección: el cambio real y el pronosticado se miden desde el valor real del día anterior
    direction = {}
    for name, _ in MODELS:
        hit, n, pt, p = pesaran_timmermann(test_data, preds[name], ref)
        direction[name] = {"hit": hit, "n": int(n), "pt": pt, "p": p}

    extras = {
        "sigma_train": sigma_train,
        "direction": direction,
        "stationarity": stationarity_tests(ts),
    }

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

    return metrics_df, fcs, train_series, test_series, predictions_dict, extras
