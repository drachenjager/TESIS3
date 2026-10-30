"""Pronóstico a h pasos adelante con origen móvil sobre el conjunto de prueba.

Para cada origen t del tramo de prueba (t = n_train−1, …, N−1−h) el modelo
pronostica y_{t+h} usando únicamente los valores reales observados hasta t:

* Naive: ŷ_{t+h} = y_t.
* SARIMA: ŷ_{t+h} = Z (Tʰ a_{t|t} + Σ_{j<h} Tʲ c) + d, con los estados filtrados
  a_{t|t} del modelo de espacio de estados (statsmodels) y parámetros fijos.
* Holt-Winters: ŷ_{t+h} = (ℓ_t + h b_t) · s_{t+h−m k}, k = ⌊(h−1)/m⌋ + 1, con los
  estados de nivel, tendencia y estacionalidad en t (parámetros fijos).
* Regresión Lineal, Random Forest, RNN y LSTM: recursivo, alimentando la
  ventana de rezagos con sus propias predicciones (todas las ventanas de prueba
  se procesan en lote).

Con h = 1 todas las fórmulas reproducen exactamente la evaluación a un paso.
"""

import numpy as np

HORIZONS = (1, 5, 20)


def targets(test_data, h):
    """Valores reales y_{t+h} para los orígenes t = n_train−1 … N−1−h (longitud n_test−h+1)."""
    test_data = np.asarray(test_data, dtype=float)
    return test_data[h - 1:] if len(test_data) >= h else test_data[:0]


def naive_multi(train_data, test_data, horizons=HORIZONS):
    hist = np.concatenate([np.asarray(train_data, dtype=float), np.asarray(test_data, dtype=float)])
    n, nt = len(train_data), len(test_data)
    return {h: hist[n - 1: n - 1 + nt - h + 1] for h in horizons if nt >= h}


def _ssm_matrix(obj, base_ndim):
    """Matriz del sistema como arreglo invariante en el tiempo.

    statsmodels guarda las matrices con ``base_ndim`` dimensiones cuando son
    constantes y con una dimensión extra (el tiempo, último eje) cuando varían.
    Devuelve ``None`` si la matriz realmente cambia a lo largo del tiempo.
    """
    arr = np.asarray(obj, dtype=float)
    if arr.ndim == base_ndim:
        return arr
    if arr.ndim == base_ndim + 1:
        first = arr[..., 0]
        if arr.shape[-1] > 1 and not np.allclose(arr, first[..., None]):
            return None
        return first
    return None


def sarima_multi(res_ext, n, nt, horizons=HORIZONS):
    """Pronósticos a h pasos desde cada origen del tramo de prueba, con los estados
    filtrados del resultado extendido (``res.append(test, refit=False)``)."""
    ssm = res_ext.model.ssm
    T = _ssm_matrix(ssm["transition"], 2)
    Z = _ssm_matrix(ssm["design"], 2)
    c = _ssm_matrix(ssm["state_intercept"], 1)
    d = _ssm_matrix(ssm["obs_intercept"], 1)
    if T is None or Z is None or c is None or d is None:
        return _sarima_multi_slow(res_ext, n, nt, horizons)
    k = T.shape[0]
    c = np.asarray(c, dtype=float).reshape(k)
    d = float(np.asarray(d, dtype=float).reshape(-1)[0])
    fs = np.asarray(res_ext.filter_results.filtered_state, dtype=float)      # (k, N)
    N = fs.shape[1]
    out = {}
    for h in horizons:
        if nt < h:
            continue
        Th = np.linalg.matrix_power(T, h)
        acc = np.zeros(k)
        Tj = np.eye(k)
        for _ in range(h):
            acc = acc + Tj @ c
            Tj = Tj @ T
        pred = (Z @ (Th @ fs + acc[:, None])).reshape(-1) + d          # pred[t] = ŷ_{t+h|t}
        out[h] = pred[n - 1: N - h]
    return out


def _sarima_multi_slow(res_ext, n, nt, horizons):
    """Camino lento (matrices variantes en el tiempo): predicción dinámica por origen."""
    N = n + nt
    out = {}
    for h in horizons:
        if nt < h:
            continue
        vals = []
        for t in range(n - 1, N - h):
            pm = res_ext.get_prediction(start=t + 1, end=t + h, dynamic=True).predicted_mean
            vals.append(float(np.asarray(pm)[-1]))
        out[h] = np.asarray(vals, dtype=float)
    return out


def hw_multi(fixed, seasonal, m, n, nt, horizons=HORIZONS):
    """Pronósticos a h pasos de Holt-Winters desde los estados (nivel, tendencia, estacionalidad)."""
    lvl = np.asarray(fixed.level, dtype=float)
    trd = getattr(fixed, "trend", None)
    trd = np.asarray(trd if trd is not None else fixed.slope, dtype=float)
    sea = np.asarray(fixed.season, dtype=float) if seasonal else None
    N = len(lvl)
    out = {}
    for h in horizons:
        if nt < h:
            continue
        ts_ = np.arange(n - 1, N - h)
        base = lvl[ts_] + h * trd[ts_]
        if seasonal:
            k = (h - 1) // m + 1
            idx = ts_ + h - m * k
            if idx.min() < 0:
                continue
            base = base * sea[idx]
        out[h] = base
    return out


def recursive_multi(predict_batch, series, n, lag, horizons=HORIZONS, newest_first=False):
    """Pronóstico recursivo en lote para modelos de ventana de rezagos.

    ``predict_batch(W)`` recibe una matriz (n_orígenes × lag) con las ventanas
    ordenadas de la observación más antigua a la más reciente (o al revés si
    ``newest_first``) y devuelve una predicción por fila. Las predicciones se
    reinyectan en la ventana para los pasos siguientes.
    """
    series = np.asarray(series, dtype=float)
    N = len(series)
    nt = N - n
    if nt <= 0 or n < lag:
        return {}
    origins = np.arange(n - 1, N - 1)
    W = np.stack([series[t - lag + 1: t + 1] for t in origins])
    out = {}
    h_max = max(horizons)
    for step in range(1, h_max + 1):
        p = np.asarray(predict_batch(W[:, ::-1] if newest_first else W), dtype=float).reshape(-1)
        if step in horizons and nt >= step:
            out[step] = p[: nt - step + 1]
        if step == h_max:
            break
        W = np.concatenate([W[:, 1:], p[:, None]], axis=1)
    return out
