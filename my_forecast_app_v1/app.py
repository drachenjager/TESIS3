"""Aplicación principal de Flask para pronosticar el tipo de cambio USD/MXN."""

from flask import Flask, render_template, request, send_file
import json
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import io

# Importamos las funciones de modelado que viven en el paquete ``models``
# (series de tiempo, machine learning y deep learning).
from models.ts_models import train_sarima, train_holtwinters
from models.ml_models import train_linear_regression, train_random_forest
from models.dl_models import train_rnn, train_lstm, prepare_data_dl
from models.stat_tests import diebolt_mariano

# Inicializamos la aplicación Flask
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    """
    En esta vista cargamos la página principal con el combo box para el período.
    Al hacer POST, obtenemos los datos, entrenamos y mostramos métricas.
    """
    if request.method == "POST":
        # 1. Leemos el período seleccionado en el formulario
        selected_period = request.form.get("period_select")

        # Porcentaje para testing
        test_percent = float(request.form.get("test_percent", 20))

        # 3. Obtenemos los datos de Yahoo Finance para ese período
        df = get_data_from_yahoo(period=selected_period)
        # 4. Entrenamos los modelos y obtenemos predicciones + métricas
        test_size = max(1, int(len(df) * test_percent / 100))
        (
            metrics_df,
            forecast_values,
            forecast_intervals,
            train_series,
            test_series,
            predictions_dict,
            dm_results,
        ) = train_and_evaluate_all_models(
            df, forecast_steps=test_size, test_size=test_size
        )

        # Determinar un ranking promedio considerando simultáneamente las
        # métricas de error (menor es mejor) y la de ajuste R^2 (mayor es
        # mejor). Agregamos una columna "Rank" para mostrar este orden en la
        # tabla de métricas.
        ranking = pd.DataFrame({
            "MAE": metrics_df["MAE"].rank(ascending=True),
            "RMSE": metrics_df["RMSE"].rank(ascending=True),
            "MAPE": metrics_df["MAPE"].rank(ascending=True),
            "R2": metrics_df["R^2"].rank(ascending=False),
        })
        metrics_df["Rank"] = ranking.mean(axis=1).rank(method="dense").astype(int)
        best_idx = metrics_df["Rank"].idxmin()

        format_dict = {
            "MAE": "{:.4f}",
            "RMSE": "{:.4f}",
            "MAPE": "{:.4f}%",
            "R^2": "{:.4f}",
            "Rank": "{:.0f}",
        }

        def highlight_best(row):
            return ["background-color: gold"] * len(row) if row.name == best_idx else [""] * len(row)

        dates = df["Date"].dt.strftime("%Y-%m-%d").tolist()
        # 5. Renderizamos la plantilla con los resultados
        metrics_table = (
            metrics_df.style
            .apply(highlight_best, axis=1)
            .format(format_dict)
            .hide(axis="index")
            .set_table_styles(
                [
                    {
                        "selector": "",
                        "props": [
                            ("border", "1px solid #000"),
                            ("border-collapse", "separate"),
                            ("border-spacing", "10px 0"),
                        ],
                    },
                    {
                        "selector": "th, td",
                        "props": [("border", "1px solid #000")],
                    },
                ]
            )
            .to_html(
                classes="table table-striped table-hover table-sm text-center",
                table_id="metrics-table",
            )
        )

        def format_forecast(vals, intervals):
            """Return a descriptive string with forecast values without confidence intervals."""
            formatted_parts = []
            for val in vals:
                if val is None or pd.isna(val):
                    continue
                formatted_parts.append(f"{float(val):.2f}")
            return "; ".join(formatted_parts) if formatted_parts else "Sin datos disponibles"

        formatted_forecasts = {
            model: format_forecast(vals, forecast_intervals.get(model))
            for model, vals in forecast_values.items()
        }

        period_labels = {
            "5d": "5 días",
            "1mo": "1 mes",
            "3mo": "3 meses",
            "6mo": "6 meses",
            "1y": "1 año",
            "ytd": "Año en curso",
            "2y": "2 años",
            "5y": "5 años",
            "10y": "10 años",
            "max": "Máx",
        }
        display_period = period_labels.get(selected_period, selected_period)

        return render_template(
            "index.html",
            metrics_table=metrics_table,
            forecast_values=formatted_forecasts,
            train_series=train_series,
            test_series=test_series,
            predictions_dict=predictions_dict,
            dm_results=dm_results,
            forecast_intervals=forecast_intervals,
            dates=dates,
            selected_period=None,
            selected_test_percent="",
            display_period=display_period,
            display_test_percent=test_percent,
        )
    else:
        # Método GET: mostramos el formulario con valores por defecto
        return render_template(
            "index.html",
            selected_period="1mo",
            selected_test_percent=20,
            dm_results=None,
            forecast_values=None,
            forecast_intervals=None,
        )


def get_data_from_yahoo(period="1y"):
    """Descarga datos históricos de Yahoo Finance.

    Parameters
    ----------
    period: str, opcional
        Periodo de consulta aceptado por yfinance ("1y" por defecto).

    Returns
    -------
    pandas.DataFrame
        DataFrame con dos columnas: ``Date`` y ``Close``.
    """
    ticker = "MXN=X"  # El par USD/MXN en Yahoo Finance se identifica como "MXN=X"
    data = yf.download(ticker, period=period, interval="1d")

    # yfinance>=0.2 puede devolver columnas MultiIndex incluso para un solo ticker
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].iloc[:, 0]
    else:
        close = data["Close"]

    # Limpieza básica: eliminamos valores faltantes y estandarizamos nombres
    close = close.dropna().reset_index()
    close.columns = ["Date", "Close"]
    return close


def train_and_evaluate_all_models(df, forecast_steps=1, test_size=5):
    """Entrena todos los modelos y calcula métricas.

    Parameters
    ----------
    df : pandas.DataFrame
        Serie temporal con columnas ``Date`` y ``Close``.
    forecast_steps : int, opcional
        Número de pasos a pronosticar hacia adelante.
    test_size : int, opcional
        Número de observaciones destinadas al conjunto de prueba.

    Returns
    -------
    tuple
        metrics_df : pandas.DataFrame
            Tabla con las métricas de evaluación de cada modelo.
        forecast_values : dict
            Pronósticos futuros por modelo.
        forecast_intervals : dict
            Intervalos de confianza (al 95%) asociados a cada pronóstico.
        train_series : list
            Serie extendida con valores de entrenamiento.
        test_series : list
            Serie extendida con valores reales del conjunto de prueba.
        predictions_dict : dict
            Predicciones alineadas temporalmente para cada modelo.
        dm_results : dict
            Resultados de la prueba Diebold-Mariano entre pares de modelos.
    """

    ts = df["Close"].values
    # Ajustamos el tamaño de prueba si es mayor que la serie
    if test_size >= len(ts):
        test_size = max(1, len(ts) // 2)
    # Separamos en entrenamiento y prueba
    train_data = ts[:-test_size]
    test_data = ts[-test_size:]

    # ----- Modelos de series de tiempo -----
    sarima_metrics, sarima_pred, sarima_forecast, sarima_ci = train_sarima(
        train_data, test_data, forecast_steps
    )
    hw_metrics, hw_pred, hw_forecast, hw_ci = train_holtwinters(
        train_data, test_data, forecast_steps
    )

    # ----- Modelos de Machine Learning -----
    linreg_metrics, linreg_pred, linreg_forecast, linreg_ci = train_linear_regression(
        train_data, test_data, forecast_steps
    )
    rf_metrics, rf_pred, rf_forecast, rf_ci = train_random_forest(
        train_data, test_data, forecast_steps
    )

    # ----- Modelos de Deep Learning -----
    rnn_metrics, rnn_pred, rnn_forecast, rnn_ci = train_rnn(
        train_data, test_data, forecast_steps
    )
    lstm_metrics, lstm_pred, lstm_forecast, lstm_ci = train_lstm(
        train_data, test_data, forecast_steps
    )

    # Construimos un DataFrame con todas las métricas obtenidas
    metrics_df = pd.DataFrame(
        [
            sarima_metrics,
            hw_metrics,
            linreg_metrics,
            rf_metrics,
            rnn_metrics,
            lstm_metrics,
        ]
    )

    # Expresar el MAPE como porcentaje para facilitar la interpretación
    metrics_df["MAPE"] = metrics_df["MAPE"] * 100

    # Diccionario con los valores de pronóstico para cada modelo
    forecast_values = {
        "SARIMA": sarima_forecast,
        "Holt-Winters": hw_forecast,
        "Regresión Lineal": linreg_forecast,
        "Random Forest": rf_forecast,
        "RNN": rnn_forecast,
        "LSTM": lstm_forecast,
    }

    forecast_intervals = {
        "SARIMA": sarima_ci,
        "Holt-Winters": hw_ci,
        "Regresión Lineal": linreg_ci,
        "Random Forest": rf_ci,
        "RNN": rnn_ci,
        "LSTM": lstm_ci,
    }

    # Listas combinando valores de train/test para facilitar el graficado
    train_series = train_data.tolist() + [None] * len(test_data)
    test_series = [None] * len(train_data) + test_data.tolist()

    # Predicciones alineadas con el eje temporal para cada modelo
    predictions_dict = {
        "SARIMA": [None] * len(train_data) + sarima_pred.tolist(),
        "Holt-Winters": [None] * len(train_data) + hw_pred.tolist(),
        "Regresión Lineal": [None] * len(train_data) + linreg_pred.tolist(),
        "Random Forest": [None] * len(train_data) + rf_pred.tolist(),
        "RNN": [None] * len(train_data) + rnn_pred.tolist(),
        "LSTM": [None] * len(train_data) + lstm_pred.tolist(),
    }

    # Resultados de la prueba Diebold-Mariano entre pares de modelos
    dm_results = {}
    model_preds = {
        "SARIMA": sarima_pred,
        "Holt-Winters": hw_pred,
        "Regresión Lineal": linreg_pred,
        "Random Forest": rf_pred,
        "RNN": rnn_pred,
        "LSTM": lstm_pred,
    }
    model_names = list(model_preds.keys())
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m1, m2 = model_names[i], model_names[j]
            stat, p_val = diebolt_mariano(test_data, model_preds[m1], model_preds[m2])
            dm_results[f"{m1} vs {m2}"] = {
                "statistic": float(stat),
                "p_value": float(p_val),
            }

    return (
        metrics_df,
        forecast_values,
        forecast_intervals,
        train_series,
        test_series,
        predictions_dict,
        dm_results,
    )


@app.route("/plot", methods=["POST"])
def plot():
    """Muestra la gráfica de un modelo seleccionado.

    Recupera las series enviadas por el formulario, las formatea para
    visualización y renderiza la plantilla ``plot.html``.
    """
    model_name = request.form.get("model_choice")
    train_series = json.loads(request.form.get("train_series"))
    test_series = json.loads(request.form.get("test_series"))
    dates = json.loads(request.form.get("dates"))
    pred_series = json.loads(request.form.get(f"pred_{model_name}"))
    raw_intervals = request.form.get("forecast_intervals")
    forecast_intervals = json.loads(raw_intervals) if raw_intervals else {}
    ci_bounds = forecast_intervals.get(model_name, {}) if forecast_intervals else {}

    ci_lower = ci_bounds.get("lower") if isinstance(ci_bounds, dict) else None
    ci_upper = ci_bounds.get("upper") if isinstance(ci_bounds, dict) else None

    test_length = sum(1 for value in test_series if value is not None)
    total_length = len(test_series)
    padding = max(0, total_length - test_length)

    def align_intervals(bounds):
        if not isinstance(bounds, list) or len(bounds) == 0:
            return [None] * total_length
        trimmed = bounds[-test_length:] if test_length > 0 else []
        if len(trimmed) < test_length:
            trimmed = [None] * (test_length - len(trimmed)) + trimmed
        return [None] * padding + trimmed

    ci_lower_aligned = align_intervals(ci_lower)
    ci_upper_aligned = align_intervals(ci_upper)
    has_confidence_data = any(v is not None for v in ci_lower_aligned + ci_upper_aligned)

    def format_series(series):
        """Convierte una serie numérica a una cadena separada por comas."""
        return ", ".join(f"{x:.2f}" for x in series if x is not None)

    # Versiones en texto de las series para mostrarlas en pantalla
    train_display = format_series(train_series)
    test_display = format_series(test_series)
    pred_display = format_series(pred_series)

    return render_template(
        "plot.html",
        model_name=model_name,
        train_series=train_series,
        test_series=test_series,
        pred_series=pred_series,
        dates=dates,
        train_display=train_display,
        test_display=test_display,
        pred_display=pred_display,
        ci_lower=ci_lower_aligned,
        ci_upper=ci_upper_aligned,
        has_confidence_data=has_confidence_data,
    )


@app.route("/download_excel", methods=["POST"])
def download_excel():
    """Genera un archivo Excel con los datos y el pronóstico del modelo.

    El archivo contiene además una gráfica lineal que compara los valores de
    entrenamiento, las observaciones reales y las predicciones.
    """
    model_name = request.form.get("model_name")
    train_series = json.loads(request.form.get("train_series"))
    test_series = json.loads(request.form.get("test_series"))
    pred_series = json.loads(request.form.get("pred_series"))
    dates = json.loads(request.form.get("dates"))

    # Construimos un DataFrame para exportar a Excel
    df = pd.DataFrame(
        {
            "Date": dates,
            "Entrenamiento": train_series,
            "Real (test)": test_series,
            "Pronóstico": pred_series,
        }
    )

    # Creamos el archivo Excel en memoria y añadimos una gráfica
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Datos")
        workbook = writer.book
        worksheet = writer.sheets["Datos"]
        chart = workbook.add_chart({"type": "line"})
        for idx, col in enumerate(["Entrenamiento", "Real (test)", "Pronóstico"]):
            chart.add_series(
                {
                    "name": col,
                    "categories": ["Datos", 1, 0, len(df), 0],
                    "values": ["Datos", 1, idx + 1, len(df), idx + 1],
                }
            )
        worksheet.insert_chart("F2", chart)
    output.seek(0)
    filename = f"{model_name}_resultado.xlsx"
    return send_file(
        output,
        as_attachment=True,
        download_name=filename,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    # app.run(debug=True)  # Para desarrollo local
    app.run(host="0.0.0.0", port=8080)  # Ajusta el puerto si es necesario
