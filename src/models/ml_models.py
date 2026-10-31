"""Modelos clásicos de Machine Learning para series temporales (sin cambios
respecto al Apéndice A: ya evaluaban a un paso adelante con el valor real)."""

import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

_ML_LAGS = 5


def _effective_lags(train_data):
    return max(1, min(_ML_LAGS, len(train_data) // 4))


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


def create_supervised_data(series, lags=5):
    df = pd.DataFrame(series, columns=["y"])
    for i in range(1, lags + 1):
        df[f"lag_{i}"] = df["y"].shift(i)
    df.dropna(inplace=True)
    X = df[[f"lag_{i}" for i in range(1, lags + 1)]].values
    y = df["y"].values
    return X, y


def _recursive_predict(model, seed_history, n_steps, lags):
    history = list(seed_history)
    preds = []
    for _ in range(n_steps):
        x = np.array([history[-i] for i in range(1, lags + 1)]).reshape(1, -1)
        p = float(model.predict(x)[0])
        preds.append(p)
        history.append(p)
    return preds


def _fit_predict(model, name, train_data, test_data, forecast_steps):
    lags = _effective_lags(train_data)
    X, y = create_supervised_data(train_data, lags=lags)
    model.fit(X, y)
    history = list(train_data)
    test_preds = []
    for actual in test_data:
        x = np.array([history[-i] for i in range(1, lags + 1)]).reshape(1, -1)
        test_preds.append(float(model.predict(x)[0]))
        history.append(actual)
    test_preds = np.array(test_preds)
    forecast_next = _recursive_predict(model, list(train_data) + list(test_data), forecast_steps, lags)
    metrics = _compute_metrics(test_preds, test_data, name)
    metrics["sigma_train"] = float(np.std(model.predict(X) - y)) if len(y) >= 3 else float("nan")
    return metrics, test_preds, forecast_next


def train_linear_regression(train_data, test_data, forecast_steps=1):
    return _fit_predict(LinearRegression(), "Regresión Lineal", train_data, test_data, forecast_steps)


def train_random_forest(train_data, test_data, forecast_steps=1):
    return _fit_predict(RandomForestRegressor(n_estimators=100, random_state=42),
                        "Random Forest", train_data, test_data, forecast_steps)
