"""Sello de procedencia: qué datos, qué versiones y qué parámetros produjeron cada resultado.

Dos corridas que difieren deben poder explicarse: distinta serie (hash), distinta
versión de una librería, distinta semilla o distinto commit del código.
"""

import hashlib
import os
import platform
import subprocess
from datetime import datetime, timezone

import numpy
import pandas
import scipy
import sklearn
import statsmodels

from src.version import APP_VERSION

try:
    import pmdarima
    _PMDARIMA = pmdarima.__version__
except ImportError:                                     # pragma: no cover
    _PMDARIMA = "no disponible"
try:
    import torch
    _TORCH = torch.__version__
except ImportError:                                     # pragma: no cover
    _TORCH = "no disponible"
try:
    import yfinance
    _YFINANCE = yfinance.__version__
except ImportError:                                     # pragma: no cover
    _YFINANCE = "no disponible"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HYPERPARAMS = [
    ("Naive",            "Paseo aleatorio sin deriva; sin parámetros."),
    ("SARIMA",           "auto_arima (pmdarima): p,q ≤ 3; P,Q ≤ 1; m = 5; criterio AIC; búsqueda stepwise."),
    ("Holt-Winters",     "Tendencia aditiva; estacionalidad multiplicativa de periodo 12 (si n ≥ 24); parámetros optimizados en entrenamiento."),
    ("Regresión Lineal", "Mínimos cuadrados con 5 rezagos (mín(5, n/4))."),
    ("Random Forest",    "100 árboles; 5 rezagos; random_state = 42."),
    ("RNN",              "50 unidades tanh; ventana 20; Adam; MSE; 50 épocas; lote 16; MinMaxScaler ajustado en entrenamiento."),
    ("LSTM",             "50 unidades; ventana 20; Adam; MSE; 50 épocas; lote 16; MinMaxScaler ajustado en entrenamiento."),
]

PROTOCOL = ("Parámetros estimados solo con el conjunto de entrenamiento. Sobre el conjunto de prueba, "
            "pronóstico con origen móvil a h = 1, 5 y 20 días: cada pronóstico usa únicamente los valores "
            "reales observados hasta el origen (SARIMA por su representación de espacio de estados, "
            "Holt-Winters por sus ecuaciones de estado, ML/DL de forma recursiva). Cada modelo se contrasta "
            "contra el modelo ingenuo con la prueba de Diebold-Mariano (corrección HLN) al 5 %.")


def _git_commit():
    env = os.environ.get("GIT_COMMIT", "").strip()
    if env:
        return env
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=3)
        return out.stdout.strip() or "n/d"
    except Exception:
        return "n/d"


GIT_COMMIT = _git_commit()
BUILD_TIME = os.environ.get("BUILD_TIME", "n/d")

VERSIONS = {
    "Python":       platform.python_version(),
    "numpy":        numpy.__version__,
    "pandas":       pandas.__version__,
    "scipy":        scipy.__version__,
    "scikit-learn": sklearn.__version__,
    "statsmodels":  statsmodels.__version__,
    "pmdarima":     _PMDARIMA,
    "torch":        _TORCH,
    "yfinance":     _YFINANCE,
}


def series_hash(df):
    """SHA-256 de la serie (fecha y cierre a 6 decimales): identifica los datos usados."""
    lines = "\n".join(f"{d.strftime('%Y-%m-%d')},{float(c):.6f}" for d, c in zip(df["Date"], df["Close"]))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def build_provenance(df, source_meta, test_size, seed=0, horizons=(1, 5, 20)):
    """Diccionario con todo lo necesario para reproducir (o explicar) un resultado."""
    n = len(df)
    sha = series_hash(df)
    return {
        "app_version":  APP_VERSION,
        "git_commit":   GIT_COMMIT,
        "build_time":   BUILD_TIME,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "source":       source_meta.get("source", "yahoo"),
        "source_label": source_meta.get("label", ""),
        "downloaded_at": source_meta.get("downloaded_at", ""),
        "source_note":  source_meta.get("note", ""),
        "ticker":       "USDMXN=X",
        "n":            n,
        "n_train":      n - test_size,
        "n_test":       test_size,
        "first_date":   df["Date"].iloc[0].strftime("%Y-%m-%d") if n else "",
        "last_date":    df["Date"].iloc[-1].strftime("%Y-%m-%d") if n else "",
        "sha256":       sha,
        "sha256_short": sha[:16],
        "seed":         seed,
        "horizons":     list(horizons),
        "versions":     dict(VERSIONS),
        "hyperparams":  list(HYPERPARAMS),
        "protocol":     PROTOCOL,
    }
