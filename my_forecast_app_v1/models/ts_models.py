"""Modelos clásicos de series de tiempo."""

import numpy as np
import pandas as pd
import math
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def train_sarima(train_data, test_data, forecast_steps=1):
    """Entrena un modelo SARIMA y devuelve métricas, residuales e intervalos."""
    # Por simplicidad, usaremos un (1,1,1) y estacionalidad = 12
    # En la práctica, se deben seleccionar p,d,q y parámetros estacionales mediante búsqueda.
    if len(train_data) == 0 or len(test_data) == 0:
        metrics = {
            "Modelo": "SARIMA",
            "MAE": float("nan"),
            "RMSE": float("nan"),
            "MAPE": float("nan"),
            "R^2": float("nan"),
        }
        return metrics, np.array([]), [None] * forecast_steps, np.array([]), []
    model = SARIMAX(
        train_data,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    sarima_fit = model.fit(disp=False)

    # Predicción en el set de prueba con intervalos
    pred_res = sarima_fit.get_prediction(
        start=len(train_data), end=len(train_data) + len(test_data) - 1, dynamic=False
    )
    predictions = pred_res.predicted_mean
    # ``conf_int`` puede devolverse como DataFrame o arreglo; lo convertimos
    # a ``np.ndarray`` de forma robusta para evitar errores al acceder a
    # ``.values`` en estructuras que ya son ``ndarray``.
    conf_int = np.asarray(pred_res.conf_int())
    residuals = test_data - predictions

    # Métricas de error
    mae = np.mean(np.abs(predictions - test_data))
    rmse = math.sqrt(np.mean((predictions - test_data) ** 2))
    mape = np.mean(
        np.abs((predictions - test_data)
               / np.where(test_data == 0, np.finfo(float).eps, test_data))
    ) * 100
    r2 = (
        1
        - (np.sum((test_data - predictions) ** 2)
           / np.sum((test_data - np.mean(test_data)) ** 2))
        if np.sum((test_data - np.mean(test_data)) ** 2) != 0
        else float("nan")
    )

    # Pronóstico del siguiente punto
    forecast_next = sarima_fit.predict(
        start=len(train_data) + len(test_data),
        end=len(train_data) + len(test_data) + forecast_steps - 1,
    )

    metrics = {
        "Modelo": "SARIMA",
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4),
        "R^2": round(r2, 4),
    }

    return (
        metrics,
        predictions,
        forecast_next.tolist(),
        residuals.tolist(),
        conf_int.tolist(),
    )


def train_holtwinters(train_data, test_data, forecast_steps=1):
    """Entrena un modelo Holt-Winters y devuelve métricas y residuales."""
    if len(train_data) == 0 or len(test_data) == 0:
        metrics = {
            "Modelo": "Holt-Winters",
            "MAE": float("nan"),
            "RMSE": float("nan"),
            "MAPE": float("nan"),
            "R^2": float("nan"),
        }
        return metrics, np.array([]), [None] * forecast_steps, np.array([]), []
    # Ver cuántos datos hay en train
    n_train = len(train_data)
    # Solo usar estacionalidad si hay >= 2 ciclos de 12
    if n_train < 24:
        # O bien no usamos componente estacional
        model = ExponentialSmoothing(train_data, trend="add", seasonal=None)
    else:
        # Caso con estacionalidad multiplicativa
        model = ExponentialSmoothing(
            train_data, seasonal_periods=12, trend="add", seasonal="mul"
        )

    hw_fit = model.fit()

    # ``HoltWintersResults`` no expone ``get_prediction`` en todas las versiones
    # de statsmodels, por lo que usamos ``predict`` y construimos intervalos
    # de manera aproximada a partir de la desviación estándar de los residuales
    # de entrenamiento.
    predictions = hw_fit.predict(
        start=len(train_data), end=len(train_data) + len(test_data) - 1
    )
    # Desviación estándar de los residuales del ajuste para aproximar el IC
    resid_std = np.std(hw_fit.resid)
    conf_int = np.column_stack(
        (
            predictions - 1.96 * resid_std,
            predictions + 1.96 * resid_std,
        )
    ) if len(predictions) > 0 else np.empty((0, 2))
    residuals = test_data - predictions

    mae = np.mean(np.abs(predictions - test_data))
    rmse = math.sqrt(np.mean((predictions - test_data) ** 2))
    mape = np.mean(
        np.abs((predictions - test_data)
               / np.where(test_data == 0, np.finfo(float).eps, test_data))
    ) * 100
    r2 = (
        1
        - (np.sum((test_data - predictions) ** 2)
           / np.sum((test_data - np.mean(test_data)) ** 2))
        if np.sum((test_data - np.mean(test_data)) ** 2) != 0
        else float("nan")
    )

    forecast_next = hw_fit.predict(
        start=len(train_data) + len(test_data),
        end=len(train_data) + len(test_data) + forecast_steps - 1,
    )

    metrics = {
        "Modelo": "Holt-Winters",
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4),
        "R^2": round(r2, 4),
    }

    return (
        metrics,
        predictions,
        forecast_next.tolist(),
        residuals.tolist(),
        conf_int.tolist(),
    )
