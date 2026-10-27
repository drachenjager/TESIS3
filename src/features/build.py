"""Construcción de features y evaluación de riesgo cambiario."""

import numpy as np


def train_test_split(series, test_size):
    """Divide la serie en train/test respetando el orden temporal."""
    if test_size >= len(series):
        test_size = max(1, len(series) // 2)
    return series[:-test_size], series[-test_size:]


def compute_risk(ts_full):
    """Calcula el semáforo de riesgo cambiario basado en volatilidad anualizada.

    Returns
    -------
    dict
        Con claves: level, label, annual_vol, desc, rec.
    """
    returns = np.diff(ts_full) / np.where(ts_full[:-1] == 0, 1e-10, ts_full[:-1])
    annual_vol = float(np.std(returns) * np.sqrt(252) * 100)

    if annual_vol < 8:
        return {
            "level": "low",
            "label": "Riesgo Bajo",
            "annual_vol": round(annual_vol, 2),
            "desc": (
                "La volatilidad histórica del tipo de cambio es baja. "
                "Condiciones relativamente estables para operaciones de importación y exportación."
            ),
            "rec": "Monitoreo periódico suficiente. Coberturas opcionales para exposiciones mayores a 90 días.",
        }
    elif annual_vol < 15:
        return {
            "level": "medium",
            "label": "Riesgo Moderado",
            "annual_vol": round(annual_vol, 2),
            "desc": (
                "Volatilidad cambiaria moderada. El tipo de cambio puede presentar "
                "movimientos significativos en el corto plazo."
            ),
            "rec": (
                "Se recomienda evaluar instrumentos de cobertura (forwards, opciones) "
                "para exposiciones mayores a 30 días."
            ),
        }
    else:
        return {
            "level": "high",
            "label": "Riesgo Alto",
            "annual_vol": round(annual_vol, 2),
            "desc": (
                "Alta volatilidad cambiaria detectada. "
                "El tipo de cambio presenta movimientos amplios e impredecibles."
            ),
            "rec": (
                "Se recomienda implementar coberturas activas de manera inmediata "
                "para proteger el valor de las operaciones."
            ),
        }
