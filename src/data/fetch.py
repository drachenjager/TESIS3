"""Descarga de datos históricos USD/MXN desde Yahoo Finance.

yfinance 1.x maneja la sesión internamente con curl_cffi (impersonación Chrome).
NO se pasa session= — hacerlo lanza YFDataException en yfinance >= 1.0.
"""

import logging
import time

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

_TICKER = "USDMXN=X"


def get_usdmxn(period=None, start_date=None, end_date=None):
    """Descarga la serie de cierre USD/MXN con 3 intentos automáticos.

    Returns
    -------
    pandas.DataFrame | None
        DataFrame con columnas ``Date`` y ``Close``, o ``None`` ante error.
    """
    log.info("Descarga: %s  period=%s  start=%s  end=%s  yfinance=%s",
             _TICKER, period, start_date, end_date, yf.__version__)

    for attempt in range(3):
        try:
            # yfinance 1.x gestiona curl_cffi internamente — sin session=
            ticker = yf.Ticker(_TICKER)

            if start_date and end_date:
                raw = ticker.history(start=start_date, end=end_date, interval="1d")
            else:
                raw = ticker.history(period=period or "1y", interval="1d")

            log.info("Intento %d: shape=%s  cols=%s",
                     attempt + 1,
                     getattr(raw, "shape", "?"),
                     list(raw.columns) if raw is not None and not raw.empty else [])

            if raw is None or raw.empty:
                log.warning("Intento %d: DataFrame vacío", attempt + 1)
                if attempt < 2:
                    time.sleep(2)
                    continue
                return None

            if "Close" not in raw.columns:
                log.error("'Close' no encontrada. Columnas: %s", list(raw.columns))
                return None

            close = raw[["Close"]].copy().dropna()
            close.index.name = "Date"
            close = close.reset_index()

            close["Date"] = pd.to_datetime(close["Date"])
            if close["Date"].dt.tz is not None:
                close["Date"] = close["Date"].dt.tz_convert(None)

            log.info("OK: %d filas  %s → %s",
                     len(close), close["Date"].iloc[0].date(), close["Date"].iloc[-1].date())
            return close[["Date", "Close"]]

        except Exception as exc:
            log.error("Intento %d falló — %s: %s", attempt + 1, type(exc).__name__, exc,
                      exc_info=True)
            if attempt < 2:
                time.sleep(2)

    log.error("Todos los intentos fallaron para %s", _TICKER)
    return None
