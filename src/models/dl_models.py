"""Modelos de Deep Learning (PyTorch) para la serie USD/MXN.

Dos arquitecturas con el mismo protocolo de entrenamiento (ventana de 20
rezagos, Adam, MSE, 50 épocas, lote 16, MinMaxScaler ajustado solo en
entrenamiento, semilla controlable):

* ``LSTM``: red recurrente con compuertas de 50 unidades (Hochreiter y
  Schmidhuber, 1997), idéntica a la del Apéndice A original.
* ``N-BEATS``: Neural Basis Expansion Analysis for Time Series (Oreshkin et al.,
  2020) en su versión genérica: bloques totalmente conectados que producen un
  *backcast* (reconstrucción de la ventana) y un *forecast*; cada bloque modela
  el residuo que deja el anterior y los pronósticos se suman. Sustituye a la RNN
  simple de la versión anterior, arquitectura que la literatura reciente
  considera superada por las redes con compuertas (Joseph y Tackes, 2024).
"""

import math
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

torch.set_num_threads(1)

_DL_LAG = 20
_NBEATS_BLOCKS = 3        # bloques residuales
_NBEATS_LAYERS = 4        # capas totalmente conectadas por bloque
_NBEATS_HIDDEN = 64       # unidades por capa


def _effective_lag(train_data):
    return max(2, min(_DL_LAG, len(train_data) // 3))


def _compute_metrics(predictions, test_data, model_name):
    predictions = np.array(predictions, dtype=float)
    test_data = np.array(test_data, dtype=float)
    mae = np.mean(np.abs(predictions - test_data))
    rmse = math.sqrt(np.mean((predictions - test_data) ** 2))
    denom = np.where(test_data == 0, np.finfo(float).eps, test_data)
    mape = np.mean(np.abs((predictions - test_data) / denom)) * 100
    ss_res = np.sum((test_data - predictions) ** 2)
    ss_tot = np.sum((test_data - np.mean(test_data)) ** 2)
    r2 = (1.0 - ss_res / ss_tot) if ss_tot != 0 else float("nan")
    return {"Modelo": model_name, "MAE": round(float(mae), 4), "RMSE": round(float(rmse), 4),
            "MAPE": round(float(mape), 4), "R^2": round(float(r2), 4)}


def _empty_metrics(model_name, n_test, forecast_steps):
    metrics = {"Modelo": model_name, "MAE": float("nan"), "RMSE": float("nan"),
               "MAPE": float("nan"), "R^2": float("nan")}
    return metrics, np.full(n_test, float("nan")), [None] * forecast_steps


def prepare_data_dl(series, lag=20):
    X, y = [], []
    for i in range(lag, len(series)):
        X.append(series[i - lag:i])
        y.append(series[i])
    return np.array(X).reshape(-1, lag, 1), np.array(y)


class _LSTMNet(nn.Module):
    def __init__(self, lag=None):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):                      # x: (lote, ventana, 1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class _NBEATSBlock(nn.Module):
    """Bloque genérico de N-BEATS: pila de capas FC + proyecciones lineales de backcast y forecast."""

    def __init__(self, lag, hidden=_NBEATS_HIDDEN, n_layers=_NBEATS_LAYERS):
        super().__init__()
        layers = [nn.Linear(lag, hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        self.fc = nn.Sequential(*layers)
        self.backcast = nn.Linear(hidden, lag)
        self.forecast = nn.Linear(hidden, 1)

    def forward(self, x):                      # x: (lote, ventana)
        h = self.fc(x)
        return self.backcast(h), self.forecast(h)


class _NBEATSNet(nn.Module):
    """N-BEATS genérico con descomposición residual doble (Oreshkin et al., 2020)."""

    def __init__(self, lag, n_blocks=_NBEATS_BLOCKS):
        super().__init__()
        self.blocks = nn.ModuleList([_NBEATSBlock(lag) for _ in range(n_blocks)])

    def forward(self, x):                      # x: (lote, ventana, 1)
        residual = x.squeeze(-1)
        forecast = torch.zeros(x.shape[0], 1)
        for block in self.blocks:
            backcast, fc = block(residual)
            residual = residual - backcast     # cada bloque modela lo que dejó el anterior
            forecast = forecast + fc
        return forecast


def _fit(model, X_train, y_train, epochs=50, batch_size=16):
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    X_t = torch.FloatTensor(X_train)
    y_t = torch.FloatTensor(y_train).unsqueeze(1)
    n = len(X_t)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i: i + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(X_t[idx]), y_t[idx])
            loss.backward()
            optimizer.step()


def _predict_batch(model, windows_np):
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(windows_np)).numpy().ravel()


def _predict_and_forecast(model, train_data, test_data, forecast_steps, scaler, lag):
    """Predicción en test (un paso adelante con el valor real) y pronóstico futuro."""
    full = np.concatenate([np.asarray(train_data, dtype=float), np.asarray(test_data, dtype=float)])
    full_s = scaler.transform(full.reshape(-1, 1)).ravel()
    n = len(train_data)
    windows = np.array([full_s[t - lag:t] for t in range(n, n + len(test_data))]).reshape(-1, lag, 1)
    preds_s = _predict_batch(model, windows)
    predictions = scaler.inverse_transform(preds_s.reshape(-1, 1)).ravel()
    history = list(full_s)
    forecast_next = []
    for _ in range(forecast_steps):
        w = np.array(history[-lag:]).reshape(1, lag, 1)
        p_s = float(_predict_batch(model, w)[0])
        forecast_next.append(float(scaler.inverse_transform([[p_s]])[0][0]))
        history.append(p_s)
    return np.array(predictions), forecast_next


def _train(net_cls, name, train_data, test_data, forecast_steps, seed):
    lag = _effective_lag(train_data)
    if len(train_data) <= lag or len(test_data) == 0:
        return _empty_metrics(name, len(test_data), forecast_steps)
    torch.manual_seed(seed)
    np.random.seed(seed)
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_s = scaler.fit_transform(np.asarray(train_data, dtype=float).reshape(-1, 1)).flatten()
    X_train, y_train = prepare_data_dl(train_s, lag=lag)
    model = net_cls(lag)
    _fit(model, X_train, y_train)
    predictions, forecast_next = _predict_and_forecast(model, train_data, test_data, forecast_steps, scaler, lag)
    metrics = _compute_metrics(predictions, test_data, name)
    fit_s = _predict_batch(model, X_train)
    fit = scaler.inverse_transform(fit_s.reshape(-1, 1)).ravel()
    y_fit = scaler.inverse_transform(np.asarray(y_train).reshape(-1, 1)).ravel()
    metrics["sigma_train"] = float(np.std(fit - y_fit)) if len(y_fit) >= 3 else float("nan")
    return metrics, predictions, forecast_next


def train_nbeats(train_data, test_data, forecast_steps=1, seed=0):
    return _train(_NBEATSNet, "N-BEATS", train_data, test_data, forecast_steps, seed)


def train_lstm(train_data, test_data, forecast_steps=1, seed=0):
    return _train(_LSTMNet, "LSTM", train_data, test_data, forecast_steps, seed)
