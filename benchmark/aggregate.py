"""Agrega results/*.json y preds/*.npz en tablas (Excel + Markdown)."""
import os, sys, json, glob, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
from models.evaluation import diebold_mariano

MODELS = ["SARIMA", "Holt-Winters", "Regresión Lineal", "Random Forest", "RNN", "LSTM"]
ALL = ["Naive"] + MODELS
PER = {"P1": "3 meses", "P2": "6 meses", "P3": "1 año", "P4": "2 años", "P5": "5 años", "P6": "10 años"}
TEST = {f"T{i}": p for i, p in enumerate(range(10, 55, 5), start=1)}
METS = ["MAE", "RMSE", "MAPE", "R^2"]
SEEDS = [0, 1, 2]
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)

recs, preds = {}, {}
for f in sorted(glob.glob(os.path.join(HERE, "results", "*.json"))):
    r = json.load(open(f, encoding="utf-8"))
    recs[(r["P"], r["T"])] = r
    preds[(r["P"], r["T"])] = dict(np.load(os.path.join(HERE, "preds", f"{r['P']}_{r['T']}.npz")))
keys = [(P, T) for P in PER for T in TEST if (P, T) in recs]
print("experimentos cargados:", len(keys))
appc = json.load(open(os.path.join(HERE, "appendix_c.json"), encoding="utf-8")) if os.path.exists(os.path.join(HERE, "appendix_c.json")) else {}


def metrics_of(r, model):
    m = r["metrics"][model]
    if model in ("RNN", "LSTM"):
        df = pd.DataFrame([m[f"seed{s}"] for s in SEEDS])
        return {k: float(df[k].mean()) for k in METS}, {k: float(df[k].std(ddof=0)) for k in METS}
    return {k: float(m[k]) for k in METS}, None


def thesis_rank(mets):
    """Ranking de la tesis: promedio de rangos (MAE, RMSE, MAPE asc.; R² desc.)."""
    df = pd.DataFrame(mets).T
    ranks = pd.concat([df["MAE"].rank(), df["RMSE"].rank(), df["MAPE"].rank(), df["R^2"].rank(ascending=False)], axis=1)
    mean = ranks.mean(axis=1)
    return mean.rank(method="min").astype(int).to_dict()


def dm_vs_naive(key, model):
    y = preds[key]["y_test"]; nv = preds[key]["Naive"]
    if model in ("RNN", "LSTM"):
        res = [diebold_mariano(y, preds[key][f"{model}_seed{s}"], nv) for s in SEEDS]
        stats_ = [s for s, p in res if np.isfinite(s)]; ps = [p for s, p in res if np.isfinite(p)]
        return (float(np.mean(stats_)) if stats_ else np.nan, float(np.median(ps)) if ps else np.nan)
    return diebold_mariano(y, preds[key][model], nv)


rows, dm_rows, order_rows, dl_rows, ba_rows = [], [], [], [], []
for key in keys:
    P, T = key; r = recs[key]
    mets, stds = {}, {}
    for m in ALL:
        mets[m], stds[m] = metrics_of(r, m)
    rk6 = thesis_rank({m: mets[m] for m in MODELS})
    rk7 = thesis_rank(mets)
    el = f"{P},{T}"
    for m in ALL:
        dm_s, dm_p = dm_vs_naive(key, m) if m != "Naive" else (np.nan, np.nan)
        if m == "Naive":
            verdict = "referencia"
        elif np.isnan(dm_p):
            verdict = "n/a"
        elif dm_p < 0.05:
            verdict = "mejor que Naive" if dm_s < 0 else "peor que Naive"
        else:
            verdict = "sin diferencia"
        rows.append({"Elemento": el, "Periodo": PER[P], "Test %": TEST[T], "n": r["n"], "n_train": r["n_train"],
                     "n_test": r["n_test"], "Modelo": m, **{k: round(mets[m][k], 4) for k in METS},
                     "Rank (6 modelos)": rk6.get(m, None), "Rank (con Naive)": rk7[m],
                     "DM vs Naive": None if np.isnan(dm_s) else round(dm_s, 3),
                     "p-valor": None if np.isnan(dm_p) else round(dm_p, 4), "Veredicto DM": verdict})
        if m != "Naive":
            dm_rows.append({"Elemento": el, "Modelo": m, "DM": dm_s, "p": dm_p, "Veredicto": verdict})
    order_rows.append({"Elemento": el, "n_train": r["n_train"], "Orden auto_arima": r.get("sarima_order", "?"),
                       "t SARIMA (s)": r["tiempos"].get("SARIMA"), "t DL (s)": r["tiempos"].get("DL")})
    for m in ("RNN", "LSTM"):
        dl_rows.append({"Elemento": el, "Modelo": m, "MAE media": round(mets[m]["MAE"], 4), "MAE desv.": round(stds[m]["MAE"], 4),
                        "R² media": round(mets[m]["R^2"], 4), "R² desv.": round(stds[m]["R^2"], 4),
                        **{f"MAE s{s}": r["metrics"][m][f"seed{s}"]["MAE"] for s in SEEDS}})
    for m in ("SARIMA", "Holt-Winters"):
        ac = appc.get(el, {}).get(m, {})
        ba_rows.append({"Elemento": el, "Periodo": PER[P], "Modelo": m,
                        "MAE Apéndice C (tesis)": ac.get("MAE"), "R² Apéndice C (tesis)": ac.get("R^2"),
                        "MAE multipaso (reproducido)": r["metrics"][f"{m}_multistep"]["MAE"],
                        "R² multipaso (reproducido)": r["metrics"][f"{m}_multistep"]["R^2"],
                        "MAE un paso (corregido)": mets[m]["MAE"], "R² un paso (corregido)": mets[m]["R^2"],
                        "MAE Naive": mets["Naive"]["MAE"]})

full = pd.DataFrame(rows)
full["P"] = full["Elemento"].str[:2]

# ---- Tablas resumen ----
wins6 = full[full["Rank (6 modelos)"] == 1].groupby(["P", "Modelo"]).size().unstack(fill_value=0).reindex(index=list(PER), columns=MODELS, fill_value=0)
wins7 = full[full["Rank (con Naive)"] == 1].groupby(["P", "Modelo"]).size().unstack(fill_value=0).reindex(index=list(PER), columns=ALL, fill_value=0)
r2p = full.pivot_table(index="Modelo", columns="P", values="R^2", aggfunc="mean").reindex(index=ALL, columns=list(PER))
maep = full.pivot_table(index="Modelo", columns="P", values="MAE", aggfunc="mean").reindex(index=ALL, columns=list(PER))
glob_ = pd.DataFrame({
    "Victorias (6 modelos)": wins6.sum().reindex(ALL).fillna(0).astype(int),
    "Victorias (con Naive)": wins7.sum().reindex(ALL).fillna(0).astype(int),
    "R² global": full.groupby("Modelo")["R^2"].mean().reindex(ALL).round(3),
    "MAE global": full.groupby("Modelo")["MAE"].mean().reindex(ALL).round(4),
    "MAE mediana": full.groupby("Modelo")["MAE"].median().reindex(ALL).round(4),
    "Mejor que Naive (DM p<0.05)": full[full["Veredicto DM"] == "mejor que Naive"].groupby("Modelo").size().reindex(ALL).fillna(0).astype(int),
    "Peor que Naive (DM p<0.05)": full[full["Veredicto DM"] == "peor que Naive"].groupby("Modelo").size().reindex(ALL).fillna(0).astype(int),
    "MAE < Naive (conteo)": full.assign(nv=full["Elemento"].map(full[full["Modelo"] == "Naive"].set_index("Elemento")["MAE"]))
        .pipe(lambda d: (d["MAE"] < d["nv"]).groupby(d["Modelo"]).sum()).reindex(ALL).fillna(0).astype(int),
})
ba = pd.DataFrame(ba_rows)
ba_sum = ba.groupby(["Modelo", "Periodo"]).agg(**{
    "MAE Apéndice C": ("MAE Apéndice C (tesis)", "mean"), "MAE reproducido multipaso": ("MAE multipaso (reproducido)", "mean"),
    "MAE corregido un paso": ("MAE un paso (corregido)", "mean"), "MAE Naive": ("MAE Naive", "mean"),
    "R² Apéndice C": ("R² Apéndice C (tesis)", "mean"), "R² corregido": ("R² un paso (corregido)", "mean")}).round(4)
ba_sum = ba_sum.reindex(pd.MultiIndex.from_product([["SARIMA", "Holt-Winters"], list(PER.values())], names=["Modelo", "Periodo"]))

# fidelidad de la reproducción (modelos deterministas, protocolo original)
fid = []
for el in full["Elemento"].unique():
    for m, k in [("SARIMA", "SARIMA_multistep"), ("Holt-Winters", "Holt-Winters_multistep")]:
        ac = appc.get(el, {}).get(m, {})
        if ac:
            P, T = el.split(","); mine = recs[(P, T)]["metrics"][k]["MAE"]
            fid.append({"Elemento": el, "Modelo": m, "MAE tesis": ac["MAE"], "MAE reproducido": mine, "dif. abs.": round(abs(ac["MAE"] - mine), 4)})
fid = pd.DataFrame(fid)

with pd.ExcelWriter(os.path.join(OUT, "Apendice_C_v2_resultados.xlsx"), engine="openpyxl") as xw:
    full.drop(columns=["P"]).to_excel(xw, "Apendice_C_v2", index=False)
    glob_.to_excel(xw, "Ranking_global")
    wins6.to_excel(xw, "Rank1_por_periodo_6modelos"); wins7.to_excel(xw, "Rank1_por_periodo_conNaive")
    r2p.round(3).to_excel(xw, "R2_promedio_periodo"); maep.round(4).to_excel(xw, "MAE_promedio_periodo")
    pd.DataFrame(dm_rows).to_excel(xw, "DM_vs_Naive", index=False)
    ba.to_excel(xw, "Antes_vs_Despues", index=False); ba_sum.to_excel(xw, "Antes_vs_Despues_resumen")
    fid.to_excel(xw, "Fidelidad_reproduccion", index=False)
    pd.DataFrame(order_rows).to_excel(xw, "Ordenes_SARIMA", index=False)
    pd.DataFrame(dl_rows).to_excel(xw, "DL_semillas", index=False)
full.drop(columns=["P"]).to_csv(os.path.join(OUT, "apendice_c_v2.csv"), index=False, encoding="utf-8-sig")


def md(df, floatfmt=4):
    return df.to_markdown(floatfmt=f".{floatfmt}f")


lines = []
lines.append("## Ranking global (54 experimentos)\n" + md(glob_, 3))
lines.append("## Conteo de primeros lugares por periodo (6 modelos de la tesis)\n" + md(wins6.rename(index=PER), 0))
lines.append("## Conteo de primeros lugares por periodo (incluyendo Naive)\n" + md(wins7.rename(index=PER), 0))
lines.append("## R² promedio por periodo\n" + md(r2p.rename(columns=PER), 3))
lines.append("## MAE promedio por periodo\n" + md(maep.rename(columns=PER), 4))
lines.append("## Antes vs. después (promedio por periodo)\n" + md(ba_sum, 4))
if len(fid):
    lines.append("## Fidelidad de la reproducción (protocolo original, modelos deterministas)\n" +
                 md(fid.groupby("Modelo")["dif. abs."].agg(["count", "mean", "max"]), 4))
for metric, fmt in [("MAE", 4), ("R^2", 3)]:
    piv = full.pivot_table(index=["Elemento", "n_train", "n_test"], columns="Modelo", values=metric).reindex(columns=ALL)
    piv = piv.reindex([f"{P},{T}" for P in PER for T in TEST if (P, T) in recs], level=0)
    best = piv.idxmin(axis=1) if metric == "MAE" else piv.idxmax(axis=1)
    piv["Mejor"] = best.values
    lines.append(f"## {metric} por elemento (7 modelos)\n" + md(piv, fmt))
rk = full.pivot_table(index="Elemento", columns="Modelo", values="Rank (6 modelos)").reindex(columns=MODELS)
rk = rk.reindex([f"{P},{T}" for P in PER for T in TEST if (P, T) in recs])
lines.append("## Rank por elemento (6 modelos, criterio de la tesis)\n" + md(rk.astype("Int64"), 0))
dmv = full[full["Modelo"] != "Naive"].pivot_table(index="Elemento", columns="Modelo", values="Veredicto DM", aggfunc="first").reindex(columns=MODELS)
dmv = dmv.reindex([f"{P},{T}" for P in PER for T in TEST if (P, T) in recs])
lines.append("## Diebold-Mariano vs. Naive por elemento\n" + dmv.to_markdown())
od = pd.DataFrame(order_rows)["Orden auto_arima"].value_counts()
lines.append("## Órdenes elegidos por auto_arima (54 experimentos)\n" + od.to_markdown())
dls = pd.DataFrame(dl_rows)
lines.append("## Variabilidad entre semillas (RNN/LSTM): desviación estándar del MAE\n" +
             md(dls.groupby("Modelo")[["MAE desv.", "R² desv."]].agg(["mean", "max"]), 4))
open(os.path.join(OUT, "tablas.md"), "w", encoding="utf-8").write("\n\n".join(lines))
print("listo:", len(lines), "tablas")
