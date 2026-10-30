"""Empaqueta los resultados publicados del banco de pruebas (Capítulo 4 / Apéndice C)
en ``src/benchmark_data/resultados_tesis.json`` para la pestaña "Banco de pruebas".

Fuentes: benchmark/out/apendice_c_v2.csv (métricas, ranking, DM) y
benchmark/results/P*_T*.json (fechas, tamaños, orden auto_arima, semillas).

Uso:  python scripts/build_published.py
"""
import csv
import glob
import json
import os
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(ROOT, "benchmark")
OUT = os.path.join(ROOT, "src", "benchmark_data", "resultados_tesis.json")

PERIODS = {"P1": ("3mo", "3 meses"), "P2": ("6mo", "6 meses"), "P3": ("1y", "1 año"),
           "P4": ("2y", "2 años"), "P5": ("5y", "5 años"), "P6": ("10y", "10 años")}
TESTS = {f"T{i}": p for i, p in enumerate(range(10, 55, 5), start=1)}
MODELS = ["Naive", "SARIMA", "Holt-Winters", "Regresión Lineal", "Random Forest", "RNN", "LSTM"]


def _f(v):
    return None if v in ("", None) else float(v)


def main():
    elements = {}
    with open(os.path.join(BENCH, "out", "apendice_c_v2.csv"), encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            el = r["Elemento"]
            P, T = el.split(",")
            e = elements.setdefault(el, {
                "P": P, "T": T, "period_key": PERIODS[P][0], "periodo": PERIODS[P][1],
                "test_pct": TESTS[T], "n": int(r["n"]), "n_train": int(r["n_train"]),
                "n_test": int(r["n_test"]), "models": {},
            })
            e["models"][r["Modelo"]] = {
                "MAE": _f(r["MAE"]), "RMSE": _f(r["RMSE"]), "MAPE": _f(r["MAPE"]), "R^2": _f(r["R^2"]),
                "rank6": None if r["Rank (6 modelos)"] in ("", None) else int(float(r["Rank (6 modelos)"])),
                "rank7": int(float(r["Rank (con Naive)"])),
                "DM": _f(r["DM vs Naive"]), "p": _f(r["p-valor"]), "veredicto": r["Veredicto DM"],
            }
    for f in sorted(glob.glob(os.path.join(BENCH, "results", "P*_T*.json"))):
        rec = json.load(open(f, encoding="utf-8"))
        el = f'{rec["P"]},{rec["T"]}'
        if el not in elements:
            continue
        e = elements[el]
        e.update({"fecha_inicio": rec["fecha_inicio"], "fecha_test": rec["fecha_test"],
                  "fecha_fin": rec["fecha_fin"], "sarima_order": rec.get("sarima_order", ""),
                  "tiempo_s": rec.get("tiempos", {}).get("total")})
        for name in ("RNN", "LSTM"):
            s0 = rec["metrics"].get(name, {}).get("seed0")
            if s0:
                e["models"][name]["seed0"] = {k: s0[k] for k in ("MAE", "RMSE", "MAPE", "R^2")}
    for el, e in elements.items():
        e["models"] = [dict(Modelo=m, **e["models"][m]) for m in MODELS if m in e["models"]]
    # resumen global
    n_naive_best = sum(1 for e in elements.values() if next(m for m in e["models"] if m["Modelo"] == "Naive")["rank7"] == 1)
    better = [(el, m["Modelo"], m["p"]) for el, e in elements.items() for m in e["models"] if m["veredicto"] == "mejor que Naive"]
    worse = {m: 0 for m in MODELS if m != "Naive"}
    for e in elements.values():
        for m in e["models"]:
            if m["veredicto"] == "peor que Naive":
                worse[m["Modelo"]] += 1
    out = {
        "generated": date.today().isoformat(), "anchor": "2026-01-01",
        "protocol": ("Evaluación a un paso adelante con parámetros estimados solo en entrenamiento; Naive como "
                     "referencia; prueba de Diebold-Mariano (HLN) al 5 %; RNN y LSTM: promedio de 3 semillas (0, 1, 2)."),
        "periods": {P: {"key": k, "label": l} for P, (k, l) in PERIODS.items()},
        "tests": TESTS, "models": MODELS,
        "summary": {"n_elements": len(elements), "naive_rank1": n_naive_best,
                    "better_than_naive": [{"elemento": el, "modelo": m, "p": p} for el, m, p in better],
                    "worse_than_naive": worse},
        "elements": elements,
    }
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{OUT}: {len(elements)} elementos; Naive rank 1 en {n_naive_best}; mejor que Naive: {better}; peor: {worse}")


if __name__ == "__main__":
    main()
