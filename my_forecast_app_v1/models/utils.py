"""Utilities shared across forecasting models."""

from typing import Iterable, List, Tuple

import numpy as np
from scipy.stats import t


def residual_confidence_intervals(
    actual: Iterable[float],
    predictions: Iterable[float],
    forecast_values: Iterable[float],
    alpha: float = 0.05,
) -> Tuple[List[float], List[float]]:
    """Compute residual-based confidence intervals for forecasts.

    Parameters
    ----------
    actual : Iterable[float]
        Observed values from the test set.
    predictions : Iterable[float]
        Model predictions aligned with ``actual``.
    forecast_values : Iterable[float]
        Future forecast values produced by the model.
    alpha : float, optional
        Significance level (0.05 by default for a 95% interval).

    Returns
    -------
    tuple of list
        Two lists (lower bounds, upper bounds) of the same length as
        ``forecast_values`` containing the confidence interval limits.
    """
    forecast_list = list(forecast_values)
    n_forecast = len(forecast_list)
    if n_forecast == 0:
        return [], []

    actual_arr = np.asarray(list(actual), dtype=float)
    pred_arr = np.asarray(list(predictions), dtype=float)

    if actual_arr.size == 0 or pred_arr.size == 0:
        return [None] * n_forecast, [None] * n_forecast

    residuals = actual_arr - pred_arr
    residuals = residuals[~np.isnan(residuals)]

    if residuals.size == 0:
        return [None] * n_forecast, [None] * n_forecast

    if residuals.size > 1:
        resid_std = np.std(residuals, ddof=1)
        degrees_freedom = residuals.size - 1
        critical_value = t.ppf(1 - alpha / 2, degrees_freedom)
    else:
        resid_std = np.std(residuals)
        critical_value = 1.96  # Approximation when only one residual is available

    if np.isnan(resid_std):
        return [None] * n_forecast, [None] * n_forecast

    if resid_std == 0:
        margins = [0.0] * n_forecast
    else:
        margin = float(critical_value * resid_std)
        margins = [margin] * n_forecast

    lower_bounds: List[float] = []
    upper_bounds: List[float] = []
    for value, margin in zip(forecast_list, margins):
        if value is None or margin is None or np.isnan(value):
            lower_bounds.append(None)
            upper_bounds.append(None)
            continue
        lower_bounds.append(float(value) - margin)
        upper_bounds.append(float(value) + margin)

    return lower_bounds, upper_bounds
