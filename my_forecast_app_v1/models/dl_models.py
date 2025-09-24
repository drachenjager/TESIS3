"""Modelos de Deep Learning para la serie USD/MXN."""

import math
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    GRU,
    LSTM,
    SimpleRNN,
    RepeatVector,
    TimeDistributed,
)

from .utils import residual_confidence_intervals


def prepare_data_dl(
    series: Iterable[float],
    input_steps: int = 1,
    output_steps: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Genera conjuntos supervisados para modelos recurrentes.

    Parameters
    ----------
    series : Iterable[float]
        Serie temporal original.
    input_steps : int, optional
        Número de observaciones pasadas que formarán cada entrada del modelo.
    output_steps : int, optional
        Número de pasos a predecir por muestra (por defecto 1).

    Returns
    -------
    tuple of numpy.ndarray
        ``(X, y)`` listos para alimentar redes recurrentes. ``X`` tendrá
        dimensiones ``(n_muestras, input_steps, 1)`` y ``y`` tendrá longitud
        ``n_muestras`` cuando ``output_steps`` sea 1, o forma
        ``(n_muestras, output_steps)`` en caso contrario.
    """

    values = np.asarray(series, dtype=float)
    if values.ndim != 1:
        values = values.reshape(-1)

    if input_steps < 1:
        raise ValueError("input_steps debe ser al menos 1")
    if output_steps < 1:
        raise ValueError("output_steps debe ser al menos 1")

    X: List[np.ndarray] = []
    y: List[np.ndarray] = []
    total_length = len(values)
    max_index = total_length - input_steps - output_steps + 1
    if max_index <= 0:
        return np.empty((0, input_steps, 1)), np.empty((0, output_steps))

    for i in range(max_index):
        end_x = i + input_steps
        end_y = end_x + output_steps
        X.append(values[i:end_x])
        y.append(values[end_x:end_y])

    X_arr = np.array(X, dtype=float).reshape((-1, input_steps, 1))
    y_arr = np.array(y, dtype=float)

    if output_steps == 1:
        y_arr = y_arr.reshape(-1)

    return X_arr, y_arr


def _empty_results(model_name: str, test_len: int, forecast_steps: int):
    metrics = {
        "Modelo": model_name,
        "MAE": float("nan"),
        "RMSE": float("nan"),
        "MAPE": float("nan"),
        "R^2": float("nan"),
    }
    empty_ci = {
        "test": {
            "lower": [None] * test_len,
            "upper": [None] * test_len,
        },
        "forecast": {
            "lower": [None] * forecast_steps,
            "upper": [None] * forecast_steps,
        },
    }
    return metrics, np.array([]), [None] * forecast_steps, empty_ci


def _evaluate_predictions(
    model_name: str,
    test_data: Iterable[float],
    predictions: Iterable[float],
    forecast_sequence: Iterable[float],
):
    predictions_arr = np.array(list(predictions), dtype=float)
    forecast_list = [float(x) for x in forecast_sequence]
    test_arr = np.array(list(test_data), dtype=float)

    mae = np.mean(np.abs(predictions_arr - test_arr))
    rmse = math.sqrt(np.mean((predictions_arr - test_arr) ** 2))
    safe_denominator = np.where(test_arr == 0, np.finfo(float).eps, test_arr)
    mape = np.mean(np.abs((predictions_arr - test_arr) / safe_denominator)) * 100
    if np.sum((test_arr - np.mean(test_arr)) ** 2) != 0:
        r2 = 1 - (
            np.sum((test_arr - predictions_arr) ** 2)
            / np.sum((test_arr - np.mean(test_arr)) ** 2)
        )
    else:
        r2 = float("nan")

    pred_list = predictions_arr.tolist()
    test_lower, test_upper = residual_confidence_intervals(test_arr, pred_list, pred_list)
    forecast_lower, forecast_upper = residual_confidence_intervals(
        test_arr, pred_list, forecast_list
    )
    ci_bounds = {
        "test": {"lower": test_lower, "upper": test_upper},
        "forecast": {"lower": forecast_lower, "upper": forecast_upper},
    }

    metrics = {
        "Modelo": model_name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4),
        "R^2": round(r2, 4),
    }

    return metrics, predictions_arr, forecast_list, ci_bounds


def train_rnn(train_data, test_data, forecast_steps=1):
    """Entrena una red neuronal recurrente simple y genera pronósticos."""
    if len(train_data) <= 1 or len(test_data) == 0:
        return _empty_results("RNN", len(test_data), forecast_steps)

    X_train, y_train = prepare_data_dl(train_data)
    if X_train.size == 0:
        return _empty_results("RNN", len(test_data), forecast_steps)

    # Definimos la arquitectura de la RNN
    model = Sequential()
    model.add(SimpleRNN(50, activation="relu", input_shape=(1, 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    # Entrenamiento de la red
    model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)

    # Predicción en el conjunto de prueba, de forma recursiva
    predictions = []
    current_input = np.array([[train_data[-1]]])  # shape=(1,1)
    current_input = current_input.reshape((1, 1, 1))

    for actual_value in test_data:
        y_pred = model.predict(current_input, verbose=0)
        predictions.append(y_pred[0][0])
        # Actualizamos input con el valor real observado
        current_input = np.array([[actual_value]])
        current_input = current_input.reshape((1, 1, 1))

    predictions = np.array(predictions)

    # Forecast del siguiente punto(s)
    forecast_next = []
    next_input = np.array([[test_data[-1]]]).reshape((1, 1, 1))
    for _ in range(forecast_steps):
        next_pred = model.predict(next_input, verbose=0)[0][0]
        forecast_next.append(next_pred)
        next_input = np.array([[next_pred]]).reshape((1, 1, 1))

    return _evaluate_predictions("RNN", test_data, predictions, forecast_next)


def train_lstm(train_data, test_data, forecast_steps=1):
    """Entrena una red LSTM y devuelve pronósticos."""
    if len(train_data) <= 1 or len(test_data) == 0:
        return _empty_results("LSTM", len(test_data), forecast_steps)

    X_train, y_train = prepare_data_dl(train_data)
    if X_train.size == 0:
        return _empty_results("LSTM", len(test_data), forecast_steps)

    # Definimos la arquitectura LSTM
    model = Sequential()
    model.add(LSTM(50, activation="relu", input_shape=(1, 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)

    predictions = []
    current_input = np.array([[train_data[-1]]])  # shape=(1,1)
    current_input = current_input.reshape((1, 1, 1))

    for actual_value in test_data:
        y_pred = model.predict(current_input, verbose=0)
        predictions.append(y_pred[0][0])
        current_input = np.array([[actual_value]])
        current_input = current_input.reshape((1, 1, 1))

    predictions = np.array(predictions)

    forecast_next = []
    next_input = np.array([[test_data[-1]]]).reshape((1, 1, 1))
    for _ in range(forecast_steps):
        next_pred = model.predict(next_input, verbose=0)[0][0]
        forecast_next.append(next_pred)
        next_input = np.array([[next_pred]]).reshape((1, 1, 1))

    return _evaluate_predictions("LSTM", test_data, predictions, forecast_next)


def train_gru(train_data, test_data, forecast_steps=1):
    """Entrena una red GRU para pronósticos univariados."""

    if len(train_data) <= 1 or len(test_data) == 0:
        return _empty_results("GRU", len(test_data), forecast_steps)

    X_train, y_train = prepare_data_dl(train_data)
    if X_train.size == 0:
        return _empty_results("GRU", len(test_data), forecast_steps)

    model = Sequential()
    model.add(GRU(50, activation="relu", input_shape=(1, 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)

    predictions = []
    current_input = np.array([[train_data[-1]]]).reshape((1, 1, 1))

    for actual_value in test_data:
        y_pred = model.predict(current_input, verbose=0)
        predictions.append(y_pred[0][0])
        current_input = np.array([[actual_value]]).reshape((1, 1, 1))

    predictions = np.array(predictions)

    forecast_next = []
    next_input = np.array([[test_data[-1]]]).reshape((1, 1, 1))
    for _ in range(forecast_steps):
        next_pred = model.predict(next_input, verbose=0)[0][0]
        forecast_next.append(next_pred)
        next_input = np.array([[next_pred]]).reshape((1, 1, 1))

    return _evaluate_predictions("GRU", test_data, predictions, forecast_next)


def train_encoder_decoder(
    train_data: Iterable[float],
    test_data: Iterable[float],
    forecast_steps: int = 1,
    input_steps: int = 5,
    output_steps: int = 1,
):
    """Entrena una arquitectura encoder-decoder basada en LSTM."""

    train_data = np.asarray(train_data, dtype=float)
    test_data = np.asarray(test_data, dtype=float)

    if len(train_data) <= input_steps or len(test_data) == 0:
        return _empty_results("Encoder-Decoder", len(test_data), forecast_steps)

    X_train, y_train = prepare_data_dl(train_data, input_steps, output_steps)
    if X_train.size == 0 or y_train.size == 0:
        return _empty_results("Encoder-Decoder", len(test_data), forecast_steps)

    model = Sequential()
    model.add(LSTM(64, activation="relu", input_shape=(input_steps, 1)))
    model.add(RepeatVector(output_steps))
    model.add(LSTM(64, activation="relu", return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.compile(optimizer="adam", loss="mse")

    y_reshaped = y_train.reshape((-1, output_steps, 1))
    model.fit(X_train, y_reshaped, epochs=60, batch_size=8, verbose=0)

    history = list(train_data)
    predictions: List[float] = []

    for actual_value in test_data:
        input_window = np.array(history[-input_steps:]).reshape((1, input_steps, 1))
        seq_pred = model.predict(input_window, verbose=0)[0]
        next_value = float(seq_pred[-1][0])
        predictions.append(next_value)
        history.append(actual_value)

    predictions_arr = np.array(predictions)

    forecast_history = history.copy()
    forecast_next: List[float] = []
    for _ in range(forecast_steps):
        input_window = np.array(forecast_history[-input_steps:]).reshape((1, input_steps, 1))
        seq_pred = model.predict(input_window, verbose=0)[0]
        next_value = float(seq_pred[-1][0])
        forecast_next.append(next_value)
        forecast_history.append(next_value)

    return _evaluate_predictions("Encoder-Decoder", test_data, predictions_arr, forecast_next)
