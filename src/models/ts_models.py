"""Modelos clásicos de series de tiempo.

Protocolo de evaluación (idéntico al de los modelos ML/DL): los parámetros se
estiman únicamente con el conjunto de entrenamiento y, sobre el conjunto de
prueba, cada predicción es a un paso adelante actualizando el estado del modelo
con el valor real observado el día anterior.
"""

import numpy as np
import math
from statsmodels.tsa.holtwinters import ExponentialSmoothing

try:
    from pmdarima import auto_arima as _pm_auto_arima
    _PMDARIMA_OK = True
except ImportError:
    _PMDARIMA_OK = False


# ---------------------------------------------------------------------------
# Helpers compartidos
# ---------------------------------------------------------------------------

def _compute_metrics(predictions, test_data, model_name):
    """Calcula MAE, RMSE, MAPE (%) y R² comparando predicciones con valores reales."""
    predictions = np.array(predictions, dtype=float)
    test_data = np.array(test_data, dtype=float)
    mae = np.mean(np.abs(predictions - test_data))
    rmse = math.sqrt(np.mean((predictions - test_data) ** 2))
    denom = np.where(test_data == 0, np.finfo(float).eps, test_data)
    mape = np.mean(np.abs((predictions - test_data) / denom)) * 100  # ya en %
    ss_res = np.sum((test_data - predictions) ** 2)
    ss_tot = np.sum((test_data - np.mean(test_data)) ** 2)
    r2 = (1.0 - ss_res / ss_tot) if ss_tot != 0 else float("nan")
    return {
        "Modelo": model_name,
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "MAPE": round(float(mape), 4),
        "R^2": round(float(r2), 4),
    }


def _empty_metrics(model_name, n_test, forecast_steps):
    metrics = {
        "Modelo": model_name,
        "MAE": float("nan"),
        "RMSE": float("nan"),
        "MAPE": float("nan"),
        "R^2": float("nan"),
    }
    return metrics, np.full(n_test, float("nan")), [None] * forecast_steps


# ---------------------------------------------------------------------------
# Modelo ingenuo (referencia): el pronóstico de mañana es el valor de hoy
# ---------------------------------------------------------------------------

def train_naive(train_data, test_data, forecast_steps=1):
    """Pronóstico ingenuo (paseo aleatorio sin deriva), un paso adelante."""
    if len(train_data) < 1 or len(test_data) == 0:
        return _empty_metrics("Naive", len(test_data), forecast_steps)
    history = np.concatenate([np.asarray(train_data, dtype=float), np.asarray(test_data, dtype=float)])
    n = len(train_data)
    predictions = history[n - 1:n - 1 + len(test_data)]
    forecast_next = [float(history[-1])] * forecast_steps
    metrics = _compute_metrics(predictions, test_data, "Naive")
    return metrics, np.array(predictions), forecast_next


# ---------------------------------------------------------------------------
# SARIMA con auto_arima (con fallback a SARIMAX fijo si pmdarima no está)
# ---------------------------------------------------------------------------

def _sarima_one_step(res, train_data, test_data, forecast_steps):
    """Predicción a un paso sobre test con parámetros fijos (estimados en train).

    ``res.append`` incorpora las observaciones de prueba al filtro de Kalman sin
    reestimar parámetros; ``get_prediction`` sobre ese tramo devuelve, para cada
    día, el pronóstico hecho el día anterior con la información real disponible.
    """
    n = len(train_data)
    res_ext = res.append(np.asarray(test_data, dtype=float), refit=False)
    pred = res_ext.get_prediction(start=n, end=n + len(test_data) - 1)
    predictions = np.asarray(pred.predicted_mean, dtype=float)
    forecast_next = np.asarray(res_ext.forecast(steps=forecast_steps), dtype=float).tolist()
    return predictions, forecast_next


def train_sarima(train_data, test_data, forecast_steps=1):
    """Entrena SARIMA usando auto_arima (selección automática de órdenes)."""
    if len(train_data) < 5 or len(test_data) == 0:
        return _empty_metrics("SARIMA", len(test_data), forecast_steps)

    try:
        if _PMDARIMA_OK:
            seasonal = len(train_data) >= 20
            model = _pm_auto_arima(
                train_data,
                max_p=3, max_q=3,
                max_P=1, max_Q=1,
                seasonal=seasonal,
                m=5,                        # estacionalidad semanal (días hábiles)
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
                information_criterion="aic",
            )
            res = model.arima_res_          # SARIMAXResults de statsmodels
        else:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            res = SARIMAX(
                train_data,
                order=(1, 1, 1),
                seasonal_order=(0, 0, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)

        predictions, forecast_next = _sarima_one_step(res, train_data, test_data, forecast_steps)
        metrics = _compute_metrics(predictions, test_data, "SARIMA")
        return metrics, predictions, forecast_next

    except Exception:
        return _empty_metrics("SARIMA", len(test_data), forecast_steps)


# ---------------------------------------------------------------------------
# Holt-Winters
# ---------------------------------------------------------------------------

def train_holtwinters(train_data, test_data, forecast_steps=1):
    """Entrena un modelo Holt-Winters y devuelve métricas y pronósticos."""
    if len(train_data) < 4 or len(test_data) == 0:
        return _empty_metrics("Holt-Winters", len(test_data), forecast_steps)

    try:
        seasonal = len(train_data) >= 24
        spec = dict(trend="add", seasonal="mul" if seasonal else None,
                    seasonal_periods=12 if seasonal else None)

        hw_fit = ExponentialSmoothing(train_data, **spec).fit(optimized=True)
        p = hw_fit.params
        n = len(train_data)

        # Misma especificación sobre train+test con TODOS los parámetros y estados
        # iniciales fijados en los valores estimados en train: los valores ajustados
        # del tramo de prueba son pronósticos a un paso adelante.
        init = dict(initialization_method="known",
                    initial_level=p["initial_level"], initial_trend=p["initial_trend"])
        if seasonal:
            init["initial_seasonal"] = p["initial_seasons"]
        full = np.concatenate([np.asarray(train_data, dtype=float), np.asarray(test_data, dtype=float)])
        fixed = ExponentialSmoothing(full, **spec, **init).fit(
            smoothing_level=p["smoothing_level"],
            smoothing_trend=p["smoothing_trend"],
            smoothing_seasonal=p["smoothing_seasonal"] if seasonal else None,
            optimized=False,
        )
        predictions = np.asarray(fixed.fittedvalues, dtype=float)[n:]
        forecast_next = np.asarray(fixed.forecast(forecast_steps), dtype=float).tolist()

        metrics = _compute_metrics(predictions, test_data, "Holt-Winters")
        return metrics, predictions, forecast_next

    except Exception:
        return _empty_metrics("Holt-Winters", len(test_data), forecast_steps)
