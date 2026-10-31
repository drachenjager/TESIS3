"""Modelos de Deep Learning (PyTorch) para la serie USD/MXN.

Idéntico al Apéndice A (RNN/LSTM de 50 unidades, Adam, MSE, 50 épocas, batch
16, MinMaxScaler ajustado solo en train). Único cambio: semilla controlable
para que las corridas sean reproducibles y repetibles.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

torch.set_num_threads(1)

_DL_LAG = 20


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


class _RNNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=50, batch_first=True, nonlinearity="tanh")
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


class _LSTMNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


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
    model = net_cls()
    _fit(model, X_train, y_train)
    predictions, forecast_next = _predict_and_forecast(model, train_data, test_data, forecast_steps, scaler, lag)
    return _compute_metrics(predictions, test_data, name), predictions, forecast_next


def train_rnn(train_data, test_data, forecast_steps=1, seed=0):
    return _train(_RNNNet, "RNN", train_data, test_data, forecast_steps, seed)


def train_lstm(train_data, test_data, forecast_steps=1, seed=0):
    return _train(_LSTMNet, "LSTM", train_data, test_data, forecast_steps, seed)
