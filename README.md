# tesis-forecast — Pronóstico del tipo de cambio USD/MXN

Plataforma web (Flask + PyTorch) desarrollada para la tesis de maestría
*Arquitectura aplicada para pronosticar el tipo de cambio USD/MXN: un estudio
comparativo entre métodos tradicionales, Machine Learning y Deep Learning por
horizonte temporal* (UNAM, Programa de Maestría y Doctorado en Ingeniería).

Aplicación desplegada: https://tesis-forecast-403096497449.us-central1.run.app/

## Modelos

| Familia | Modelo | Implementación |
|---|---|---|
| Referencia | Naive (paseo aleatorio) | el pronóstico de mañana es el valor de hoy |
| Series de tiempo | SARIMA | `pmdarima.auto_arima` (m = 5) |
| Series de tiempo | Holt-Winters | `statsmodels` (tendencia aditiva, estacionalidad multiplicativa) |
| Machine Learning | Regresión Lineal | 5 rezagos (`scikit-learn`) |
| Machine Learning | Random Forest | 5 rezagos, 100 árboles |
| Deep Learning | RNN | 50 unidades, ventana 20, 50 épocas (PyTorch) |
| Deep Learning | LSTM | 50 unidades, ventana 20, 50 épocas (PyTorch) |

## Protocolo de evaluación (v2)

Todos los modelos se evalúan de la misma forma:

1. Los parámetros se estiman **solo con el conjunto de entrenamiento**.
2. Sobre el conjunto de prueba cada predicción es **a un paso adelante**: el
   modelo actualiza su estado con el valor real del día anterior
   (`src/models/ts_models.py`, `ml_models.py`, `dl_models.py`).
3. Métricas: MAE, RMSE, MAPE y R². Ranking compuesto por promedio de rangos
   (`src/models/ranking.py`).
4. Cada modelo se contrasta contra Naive con la prueba de **Diebold-Mariano**
   (corrección de Harvey-Leybourne-Newbold; `src/models/evaluation.py`).
5. Las redes neuronales usan semilla fija (`seed=0`) para ser reproducibles.

En la versión anterior (v1) SARIMA y Holt-Winters pronosticaban toda la ventana
de prueba de una vez, lo que los hacía incomparables con el resto; el cambio de
protocolo y sus efectos están documentados en `benchmark/`.

## Banco de Tesis (`/banco`)

Página de solo lectura con los 54 experimentos publicados en el Capítulo 4 y el
Apéndice C (mapa de resultados, conteos de primeros lugares, veredicto de
Diebold-Mariano por modelo y el detalle de cada experimento). Los datos se leen
de `src/benchmark_data/resultados_tesis.json`, que genera
`scripts/build_published.py` a partir de `benchmark/`; en esa página no se
entrena nada.

## Ejecución local

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python wsgi.py                                      # http://localhost:8080
```

Modos de consulta: periodo relativo a hoy, fecha ancla + periodo (p. ej.
2026-01-01 + 1 año reproduce los experimentos de la tesis) y rango de fechas.

## Despliegue (Cloud Run)

```bash
make deploy
```

Equivale a `gcloud run deploy tesis-forecast --source . --project tesis-usdmxn-2026
--region us-central1 --memory 2Gi --cpu 1 --timeout 300 --session-affinity ...`
(ver `Makefile`). La imagen se construye con `Dockerfile` (torch CPU-only).

## Reproducir el banco de pruebas de la tesis (54 experimentos)

`benchmark/` contiene la matriz completa: 6 longitudes de historia (3 meses a
10 años, ancladas al 2026-01-01) × 9 particiones de prueba (10 % a 50 %).

```bash
python benchmark/run_bench.py P1      # ... hasta P6 (o benchmark/launch.ps1 en paralelo)
python benchmark/aggregate.py         # tablas en benchmark/out/
```

Salidas incluidas: `results/*.json` (métricas por experimento, 3 semillas en
RNN/LSTM), `preds/*.npz` (predicciones), `out/Apendice_C_v2_resultados.xlsx`
y `out/tablas.md`. `benchmark/models/ts_models.py` conserva además las
funciones `*_multistep` del protocolo v1 para la comparación "antes vs. después".

## Estructura

```
app/        Flask: rutas (main, exports), plantillas y estáticos
src/data/   descarga de datos (yfinance)
src/models/ modelos, pipeline, ranking y prueba de Diebold-Mariano
benchmark/  banco de pruebas reproducible de la tesis
scripts/    utilidades (exportar Apéndice A, favicon)
wsgi.py     punto de entrada (gunicorn / desarrollo)
```
