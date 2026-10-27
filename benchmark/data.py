"""Descarga (con caché local) de la serie USD/MXN anclada a la fecha de las corridas."""
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "usdmxn_cache.csv")
ANCHOR = datetime(2026, 1, 1)          # fecha de las ejecuciones reportadas en la tesis
OFFSETS = {"3mo": relativedelta(months=3), "6mo": relativedelta(months=6),
           "1y": relativedelta(years=1), "2y": relativedelta(years=2),
           "5y": relativedelta(years=5), "10y": relativedelta(years=10)}
PERIOD_KEYS = {"P1": "3mo", "P2": "6mo", "P3": "1y", "P4": "2y", "P5": "5y", "P6": "10y"}
TEST_PCT = {f"T{i}": p for i, p in enumerate(range(10, 55, 5), start=1)}


def load_full():
    if os.path.exists(CACHE):
        df = pd.read_csv(CACHE, parse_dates=["Date"])
        return df
    import yfinance as yf
    raw = yf.Ticker("USDMXN=X").history(start="2015-11-01", end=ANCHOR.strftime("%Y-%m-%d"), interval="1d")
    close = raw[["Close"]].dropna()
    close.index.name = "Date"
    close = close.reset_index()
    close["Date"] = pd.to_datetime(close["Date"])
    if close["Date"].dt.tz is not None:
        close["Date"] = close["Date"].dt.tz_convert(None)
    close = close[["Date", "Close"]]
    close.to_csv(CACHE, index=False)
    return close


def series_for(period_key):
    full = load_full()
    start = ANCHOR - OFFSETS[PERIOD_KEYS[period_key]]
    sub = full[(full["Date"] >= start) & (full["Date"] < ANCHOR)]
    return sub["Close"].to_numpy(dtype=float), sub["Date"].to_numpy()


def split(series, test_key):
    n = len(series)
    test_size = max(1, int(n * TEST_PCT[test_key] / 100))
    if test_size >= n:
        test_size = max(1, n // 2)
    return series[:-test_size], series[-test_size:]
