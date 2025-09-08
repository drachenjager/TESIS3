"""Pruebas estadísticas para comparar modelos de pronóstico."""

import numpy as np
from scipy.stats import t


def diebolt_mariano(actual, pred1, pred2, h=1, loss="MSE", alternative="two-sided"):
    """Calcula el estadístico y el p-valor de la prueba Diebold-Mariano.

    Parameters
    ----------
    actual : array-like
        Observaciones reales.
    pred1 : array-like
        Predicciones del modelo 1.
    pred2 : array-like
        Predicciones del modelo 2.
    h : int, opcional
        Horizonte de pronóstico utilizado por los modelos (1 por defecto).
    loss : {"MSE", "MAE"}, opcional
        Función de pérdida para calcular las diferencias de error.
    alternative : {"two-sided", "less", "greater"}, opcional
        Tipo de hipótesis alternativa para el cálculo del p-valor.

    Returns
    -------
    tuple
        ``(dm_statistic, p_value)`` donde ``dm_statistic`` es el valor del
        estadístico DM y ``p_value`` el p-valor asociado.
    """
    actual = np.asarray(actual)
    pred1 = np.asarray(pred1)
    pred2 = np.asarray(pred2)

    e1 = actual - pred1
    e2 = actual - pred2

    loss = loss.upper()
    if loss == "MSE":
        d = e1 ** 2 - e2 ** 2
    elif loss in {"MAE", "MAD"}:
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("Loss no soportada: use 'MSE' o 'MAE'.")

    d_mean = np.mean(d)
    n = len(d)

    def autocov(x, lag):
        return np.sum((x[lag:] - d_mean) * (x[: n - lag] - d_mean)) / n

    gamma0 = autocov(d, 0)
    var_d = gamma0
    for lag in range(1, h):
        gamma = autocov(d, lag)
        weight = 1 - lag / h
        var_d += 2 * weight * gamma

    dm_stat = d_mean / np.sqrt(var_d / n)

    df = n - 1
    if alternative == "two-sided":
        p_value = 2 * (1 - t.cdf(np.abs(dm_stat), df))
    elif alternative == "less":
        p_value = t.cdf(dm_stat, df)
    elif alternative == "greater":
        p_value = 1 - t.cdf(dm_stat, df)
    else:
        raise ValueError("Valor de 'alternative' no válido.")

    return dm_stat, p_value
