"""Comparación estadística de pronósticos y diagnósticos de la serie.

* ``diebold_mariano``: ¿un modelo tiene menor error esperado que otro?
* ``holm_adjust``: corrección de Holm para varias comparaciones a la vez.
* ``pesaran_timmermann``: ¿el modelo acierta la dirección (sube/baja) más que el azar?
* ``ljung_box_p``: ¿los residuales conservan autocorrelación o ya son ruido?
* ``stationarity_tests``: ADF y KPSS de la serie en nivel y en primeras diferencias.
"""

import warnings

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


def holm_adjust(pvalues):
    """Corrección de Holm (1979) para ``m`` comparaciones simultáneas.

    Recibe una lista de p-valores (los NaN se ignoran y se devuelven como NaN) y
    devuelve los p-valores ajustados: se ordenan de menor a mayor, el i-ésimo se
    multiplica por (m − i + 1) y se fuerza a que la secuencia no decrezca. Un
    p ajustado < 0.05 sigue siendo significativo aunque se hagan m pruebas.
    """
    p = np.asarray(pvalues, dtype=float)
    out = np.full(len(p), np.nan)
    idx = np.where(np.isfinite(p))[0]
    m = len(idx)
    if m == 0:
        return out.tolist()
    order = idx[np.argsort(p[idx])]
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * p[i])
        running = max(running, adj)
        out[i] = running
    return out.tolist()


def pesaran_timmermann(y_true, y_pred, y_prev):
    """Acierto de dirección y prueba de Pesaran-Timmermann (1992).

    Compara el signo del cambio real (y_t − y_{t−1}) con el del cambio pronosticado
    (ŷ_t − y_{t−1}). Devuelve ``(acierto, n, estadístico, p)``: ``acierto`` es la
    fracción de días con la dirección correcta, ``n`` los días evaluados (se
    excluyen los días sin cambio real o sin dirección pronosticada), ``p`` es el
    p-valor unilateral de H1: el acierto supera al esperado por azar. Si quedan
    menos de 10 días o el pronóstico nunca toma dirección (Naive), devuelve NaN.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_prev = np.asarray(y_prev, dtype=float)
    da = np.sign(y_true - y_prev)
    dp = np.sign(y_pred - y_prev)
    mask = np.isfinite(da) & np.isfinite(dp) & (da != 0) & (dp != 0)
    da, dp = da[mask], dp[mask]
    n = len(da)
    if n < 10:
        return float("nan"), int(n), float("nan"), float("nan")
    hit = float(np.mean(da == dp))
    py = float(np.mean(da > 0))
    px = float(np.mean(dp > 0))
    p_star = py * px + (1 - py) * (1 - px)
    var_p = p_star * (1 - p_star) / n
    var_star = ((2 * py - 1) ** 2 * px * (1 - px) / n + (2 * px - 1) ** 2 * py * (1 - py) / n
                + 4 * py * px * (1 - py) * (1 - px) / n ** 2)
    denom = var_p - var_star
    if denom <= 0:
        return hit, n, float("nan"), float("nan")
    pt = (hit - p_star) / np.sqrt(denom)
    return hit, n, float(pt), float(stats.norm.sf(pt))


def ljung_box_p(resid, lags=10):
    """p-valor de la prueba de Ljung-Box sobre los residuales (H0: sin autocorrelación
    hasta ``lags`` rezagos). p > 0.05 => lo que queda en los errores es ruido."""
    from statsmodels.stats.diagnostic import acorr_ljungbox
    r = np.asarray(resid, dtype=float)
    r = r[np.isfinite(r)]
    lags = int(max(1, min(lags, len(r) // 4)))
    if len(r) < 8:
        return float("nan"), lags
    try:
        res = acorr_ljungbox(r, lags=[lags], return_df=True)
        return float(res["lb_pvalue"].iloc[0]), lags
    except Exception:
        return float("nan"), lags


def stationarity_tests(series):
    """ADF y KPSS en nivel y en primeras diferencias.

    Devuelve un diccionario con, para ``nivel`` y ``diferencia``, el estadístico y
    el p-valor de cada prueba (KPSS reporta p acotado a [0.01, 0.10] por sus
    tablas) y una conclusión en texto.
    """
    from statsmodels.tsa.stattools import adfuller, kpss
    s = np.asarray(series, dtype=float)
    s = s[np.isfinite(s)]
    out = {}
    for name, x in (("nivel", s), ("diferencia", np.diff(s))):
        rec = {"n": int(len(x))}
        if len(x) < 20:
            rec.update(adf_stat=float("nan"), adf_p=float("nan"), kpss_stat=float("nan"), kpss_p=float("nan"))
            out[name] = rec
            continue
        try:
            adf = adfuller(x, autolag="AIC")
            rec["adf_stat"], rec["adf_p"] = float(adf[0]), float(adf[1])
        except Exception:
            rec["adf_stat"], rec["adf_p"] = float("nan"), float("nan")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kp = kpss(x, regression="c", nlags="auto")
            rec["kpss_stat"], rec["kpss_p"] = float(kp[0]), float(kp[1])
        except Exception:
            rec["kpss_stat"], rec["kpss_p"] = float("nan"), float("nan")
        out[name] = rec

    lv, df = out["nivel"], out["diferencia"]
    rw_level = (lv.get("adf_p", 1) > 0.05) and (lv.get("kpss_p", 0) < 0.05)
    stat_diff = (df.get("adf_p", 1) < 0.05) and (df.get("kpss_p", 0) >= 0.05)
    if rw_level and stat_diff:
        concl = ("La serie no es estacionaria en nivel (ADF no rechaza la raíz unitaria y KPSS rechaza la "
                 "estacionariedad) pero sí en primeras diferencias: se comporta como un paseo aleatorio, lo que "
                 "justifica usar el modelo ingenuo como referencia.")
    elif rw_level:
        concl = ("La serie no es estacionaria en nivel; en primeras diferencias las dos pruebas no coinciden del "
                 "todo, aunque el comportamiento es cercano al de un paseo aleatorio.")
    elif lv.get("adf_p", 1) <= 0.05 and lv.get("kpss_p", 0) >= 0.05:
        concl = "Las pruebas indican que la serie es estacionaria en nivel: no se comporta como paseo aleatorio en esta ventana."
    else:
        concl = "Las pruebas ADF y KPSS no coinciden en nivel; la evidencia sobre la raíz unitaria es ambigua en esta ventana."
    out["conclusion"] = concl
    out["random_walk"] = bool(rw_level)
    return out
