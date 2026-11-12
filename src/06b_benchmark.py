import json
import statistics
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

PANEL = "data_processed/panel_clean.parquet"
COLS = ["CERT", "QDATE", "ASSET", "EQ", "LNLSGR", "LNATRES", "NCLNLS", "ROA", "y"]
DEV_END = date(2009, 12, 31)
REPEAT = 3


def run_pandas():
    import pandas as pd
    df = pd.read_parquet(PANEL, columns=COLS)
    df["QDATE"] = pd.to_datetime(df["QDATE"])
    df = df[df["QDATE"] <= pd.Timestamp(DEV_END)].copy()
    df["EQ_RATIO"] = df["EQ"] / df["ASSET"] * 100
    df["NPL_RATIO"] = df["NCLNLS"] / df["LNLSGR"].where(df["LNLSGR"] != 0) * 100
    df["RES_RATIO"] = df["LNATRES"] / df["LNLSGR"].where(df["LNLSGR"] != 0) * 100
    df = df.sort_values(["CERT", "QDATE"])
    g = df.groupby("CERT", sort=False)
    df["ASSET_G4"] = df["ASSET"] / g["ASSET"].shift(4) - 1
    df["EQ_RATIO_D4"] = df["EQ_RATIO"] - g["EQ_RATIO"].shift(4)
    df["ROA_MA4"] = g["ROA"].rolling(4).mean().reset_index(level=0, drop=True)
    out = df.groupby("QDATE").agg(
        n_rows=("CERT", "size"), med_EQ_RATIO=("EQ_RATIO", "median"),
        med_NPL_RATIO=("NPL_RATIO", "median"), med_ASSET_G4=("ASSET_G4", "median"),
        med_ROA_MA4=("ROA_MA4", "median"), n_positive=("y", "sum")).sort_index()
    return out.reset_index()


def polars_query(lf):
    import polars as pl
    lf = (lf.filter(pl.col("QDATE") <= DEV_END)
          .select(COLS)
          .with_columns((pl.col("EQ") / pl.col("ASSET") * 100).alias("EQ_RATIO"),
                        (pl.col("NCLNLS") / pl.when(pl.col("LNLSGR") != 0).then(pl.col("LNLSGR")) * 100).alias("NPL_RATIO"),
                        (pl.col("LNATRES") / pl.when(pl.col("LNLSGR") != 0).then(pl.col("LNLSGR")) * 100).alias("RES_RATIO"))
          .sort("CERT", "QDATE")
          .with_columns((pl.col("ASSET") / pl.col("ASSET").shift(4).over("CERT") - 1).alias("ASSET_G4"),
                        (pl.col("EQ_RATIO") - pl.col("EQ_RATIO").shift(4).over("CERT")).alias("EQ_RATIO_D4"),
                        pl.col("ROA").rolling_mean(window_size=4).over("CERT").alias("ROA_MA4")))
    return (lf.group_by("QDATE").agg(
        pl.len().alias("n_rows"), pl.col("EQ_RATIO").median().alias("med_EQ_RATIO"),
        pl.col("NPL_RATIO").median().alias("med_NPL_RATIO"), pl.col("ASSET_G4").median().alias("med_ASSET_G4"),
        pl.col("ROA_MA4").median().alias("med_ROA_MA4"), pl.col("y").sum().alias("n_positive")).sort("QDATE"))


def run_polars_lazy():
    import polars as pl
    return polars_query(pl.scan_parquet(PANEL)).collect().to_pandas()


def run_polars_eager():
    import polars as pl
    df = pl.read_parquet(PANEL)
    return polars_query(df.lazy()).collect(optimizations=pl.QueryOptFlags.none()).to_pandas()


FUNCS = {"pandas": run_pandas, "polars_eager": run_polars_eager, "polars_lazy": run_polars_lazy}


def child(name):
    import psutil
    t0 = time.perf_counter()
    res = FUNCS[name]()
    secs = time.perf_counter() - t0
    peak = psutil.Process().memory_info().peak_wset / 2**20
    res.to_parquet(f"data_processed/bench_{name}.parquet")
    print(json.dumps({"secs": secs, "peak_mb": peak}))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        child(sys.argv[1])
        sys.exit()
    import pandas as pd
    rows = []
    for name in FUNCS:
        runs = []
        for _ in range(REPEAT):
            out = subprocess.run([sys.executable, __file__, name], capture_output=True, text=True)
            if out.returncode != 0:
                print(out.stderr[-800:])
                raise SystemExit(f"{name} failed")
            runs.append(json.loads(out.stdout.strip().splitlines()[-1]))
        rows.append({"method": name,
                     "median_seconds": round(statistics.median(r["secs"] for r in runs), 2),
                     "peak_RAM_MB": round(statistics.median(r["peak_mb"] for r in runs)),
                     "all_runs_s": [round(r["secs"], 2) for r in runs]})
        print(rows[-1])
    res = pd.DataFrame(rows)
    base = res.loc[res.method == "pandas", "median_seconds"].item()
    res["speedup_vs_pandas"] = (base / res["median_seconds"]).round(1)
    res.to_csv("outputs/part3_polars_vs_pandas.csv", index=False)
    print("\n", res.to_string(index=False))

    a = pd.read_parquet("data_processed/bench_pandas.parquet")
    for other in ("polars_eager", "polars_lazy"):
        b = pd.read_parquet(f"data_processed/bench_{other}.parquet")
        a2 = a.sort_values("QDATE").reset_index(drop=True)
        b2 = b.sort_values("QDATE").reset_index(drop=True)
        diffs = {c: float((a2[c].astype(float) - b2[c].astype(float)).abs().max())
                 for c in a2.columns if c != "QDATE"}
        print(f"max abs diff pandas vs {other}:", diffs)