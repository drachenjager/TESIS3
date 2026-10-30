"""Serie congelada de la tesis.

``usdmxn_tesis.csv`` es la serie de cierre diario USDMXN=X descargada de Yahoo
Finance el 26-oct-2026 para el banco de pruebas (``benchmark/usdmxn_cache.csv``).
Se distribuye con la plataforma para que los 54 experimentos del Capítulo 4 y del
Apéndice C puedan reproducirse aunque Yahoo revise o deje de servir los datos.
"""

import os
from datetime import datetime

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN_CSV = os.path.join(HERE, "usdmxn_tesis.csv")
FROZEN_ANCHOR = datetime(2026, 1, 1)        # fecha ancla de los 54 experimentos
FROZEN_INFO = {
    "ticker": "USDMXN=X",
    "descargada": "2026-10-26",
    "rango": "2015-11-02 → 2025-12-31",
    "descripcion": ("Serie de cierre diario USDMXN=X descargada de Yahoo Finance el 26-oct-2026 "
                    "para el banco de pruebas de la tesis (2,648 observaciones)."),
}

_FROZEN = None


def load_frozen():
    """Carga (una sola vez) la serie congelada completa."""
    global _FROZEN
    if _FROZEN is None:
        df = pd.read_csv(FROZEN_CSV, parse_dates=["Date"])
        _FROZEN = df[["Date", "Close"]].sort_values("Date").reset_index(drop=True)
    return _FROZEN.copy()


def get_frozen_usdmxn(start_date=None, end_date=None):
    """Recorte de la serie congelada: ``start_date`` inclusivo y ``end_date``
    exclusivo, igual que ``yfinance.Ticker.history(start, end)``."""
    df = load_frozen()
    if start_date:
        df = df[df["Date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["Date"] < pd.Timestamp(end_date)]
    return df.reset_index(drop=True)
