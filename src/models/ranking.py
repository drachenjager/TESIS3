"""Ranking compuesto de modelos (criterio de la tesis)."""

import pandas as pd

_ERROR_METRICS = ["MAE", "RMSE", "MAPE"]


def add_ranking(metrics_df):
    """Agrega la columna ``Rank`` y devuelve ``(df_ordenado, best_idx)``.

    Cada métrica de error se rankea ascendente y R² descendente; el promedio de
    rangos define el Rank final (``method="min"``: los empates comparten lugar).
    Los modelos con métricas NaN reciben el peor rango en vez de romper el cálculo.
    """
    df = metrics_df.copy()
    ranks = pd.DataFrame({
        m: df[m].rank(ascending=True, na_option="bottom") for m in _ERROR_METRICS
    })
    ranks["R2"] = df["R^2"].rank(ascending=False, na_option="bottom")
    df["Rank"] = ranks.mean(axis=1).rank(method="min").astype(int)
    df = df.sort_values(["Rank", "MAE"], kind="stable")
    best_idx = df["Rank"].idxmin()
    return df, best_idx
