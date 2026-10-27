"""Comparación estadística de pronósticos."""

import numpy as np
from scipy import stats


def diebold_mariano(y_true, pred_a, pred_b, h=1, loss="squared"):
    """Prueba de Diebold-Mariano (1995) con la corrección de Harvey, Leybourne y
    Newbold (1997) para muestras pequeñas.

    H0: ambos pronósticos tienen la misma precisión esperada.
    Estadístico negativo => el pronóstico ``a`` tiene menor pérdida que ``b``.
    Devuelve (estadístico, p-valor bilateral); (nan, nan) si no es calculable.
    """
    y_true = np.asarray(y_true, dtype=float)
    e_a = y_true - np.asarray(pred_a, dtype=float)
    e_b = y_true - np.asarray(pred_b, dtype=float)
    mask = np.isfinite(e_a) & np.isfinite(e_b)
    e_a, e_b = e_a[mask], e_b[mask]
    n = len(e_a)
    if n < 3:
        return float("nan"), float("nan")
    d = (e_a ** 2 - e_b ** 2) if loss == "squared" else (np.abs(e_a) - np.abs(e_b))
    d_bar = d.mean()
    dc = d - d_bar
    gamma = [np.dot(dc, dc) / n]
    for k in range(1, h):
        gamma.append(np.dot(dc[k:], dc[:-k]) / n)
    var_d = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var_d <= 0:
        return float("nan"), float("nan")
    dm = d_bar / np.sqrt(var_d)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_c = dm * hln
    p = 2 * stats.t.sf(abs(dm_c), df=n - 1)
    return float(dm_c), float(p)
