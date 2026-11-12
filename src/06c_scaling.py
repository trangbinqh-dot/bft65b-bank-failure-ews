import json
import shutil
import statistics
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

PANEL = Path("data_processed/panel_clean.parquet")
SCALE_DIR = Path("data_processed/scale_tmp")
COLS = ["CERT", "QDATE", "ASSET", "EQ", "LNLSGR", "LNATRES", "NCLNLS", "ROA", "y"]
DEV_END = date(2009, 12, 31)
SCALES = (1, 2, 3)
REPEAT = 3
RAM_SAFE_LIMIT_MB = 5800  # Dung an toan cho laptop 8 GB RAM


def make_scaled_parquet(scale: int) -> tuple[Path, int, float]:
    import polars as pl
    SCALE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SCALE_DIR / f"panel_{scale}x.parquet"
    if scale == 1:
        n_rows = pl.scan_parquet(PANEL).select(pl.len()).collect().item()
        return PANEL, n_rows, PANEL.stat().st_size / 2**20

    base = pl.read_parquet(PANEL)
    # Tao cac ban sao voi ma CERT dich chuyen de giu nguyen so quy cua moi ngan hang
    copies = [
        base.with_columns((pl.col("CERT") + k * 1_000_000).cast(pl.Int32).alias("CERT"))
        for k in range(scale)
    ]
    big = pl.concat(copies, rechunk=True)
    big.write_parquet(out_path, compression="zstd")
    return out_path, big.height, out_path.stat().st_size / 2**20


def run_pandas(path_str: str):
    import pandas as pd
    df = pd.read_parquet(path_str, columns=COLS)
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
        n_rows=("CERT", "size"),
        med_EQ_RATIO=("EQ_RATIO", "median"),
        med_NPL_RATIO=("NPL_RATIO", "median"),
        med_ASSET_G4=("ASSET_G4", "median"),
        med_ROA_MA4=("ROA_MA4", "median"),
        n_positive=("y", "sum"),
    ).sort_index()
    return len(out)


def run_polars_lazy(path_str: str):
    import polars as pl
    lf = (
        pl.scan_parquet(path_str)
        .filter(pl.col("QDATE") <= DEV_END)
        .select(COLS)
        .with_columns(
            (pl.col("EQ") / pl.col("ASSET") * 100).alias("EQ_RATIO"),
            (pl.col("NCLNLS") / pl.when(pl.col("LNLSGR") != 0).then(pl.col("LNLSGR")) * 100).alias("NPL_RATIO"),
            (pl.col("LNATRES") / pl.when(pl.col("LNLSGR") != 0).then(pl.col("LNLSGR")) * 100).alias("RES_RATIO"),
        )
        .sort("CERT", "QDATE")
        .with_columns(
            (pl.col("ASSET") / pl.col("ASSET").shift(4).over("CERT") - 1).alias("ASSET_G4"),
            (pl.col("EQ_RATIO") - pl.col("EQ_RATIO").shift(4).over("CERT")).alias("EQ_RATIO_D4"),
            pl.col("ROA").rolling_mean(window_size=4).over("CERT").alias("ROA_MA4"),
        )
        .group_by("QDATE")
        .agg(
            pl.len().alias("n_rows"),
            pl.col("EQ_RATIO").median().alias("med_EQ_RATIO"),
            pl.col("NPL_RATIO").median().alias("med_NPL_RATIO"),
            pl.col("ASSET_G4").median().alias("med_ASSET_G4"),
            pl.col("ROA_MA4").median().alias("med_ROA_MA4"),
            pl.col("y").sum().alias("n_positive"),
        )
        .sort("QDATE")
    )
    res = lf.collect()
    return res.height


FUNCS = {"pandas": run_pandas, "polars_lazy": run_polars_lazy}


def child(method: str, path_str: str):
    import psutil
    t0 = time.perf_counter()
    n_out = FUNCS[method](path_str)
    secs = time.perf_counter() - t0
    peak = psutil.Process().memory_info().peak_wset / 2**20
    print(json.dumps({"secs": secs, "peak_mb": peak, "n_out": n_out}))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        child(sys.argv[1], sys.argv[2])
        sys.exit(0)

    import pandas as pd
    Path("outputs").mkdir(exist_ok=True)
    rows = []
    skip_pandas = False

    try:
        for s in SCALES:
            fpath, total_rows, disk_mb = make_scaled_parquet(s)
            print(f"\n=== SCALE {s}x ({total_rows:,} total panel rows, {disk_mb:.1f} MB Parquet) ===")
            for method in ("polars_lazy", "pandas"):
                if method == "pandas" and skip_pandas:
                    print(f"  {method}: SKIPPED (previous scale approached 8 GB RAM limit)")
                    continue
                runs = []
                failed = False
                for _ in range(REPEAT):
                    out = subprocess.run(
                        [sys.executable, __file__, method, str(fpath)],
                        capture_output=True, text=True
                    )
                    if out.returncode != 0:
                        print(f"  {method} FAILED at {s}x:\n", out.stderr[-500:])
                        failed = True
                        break
                    runs.append(json.loads(out.stdout.strip().splitlines()[-1]))
                if failed:
                    if method == "pandas":
                        skip_pandas = True
                    continue
                med_s = round(statistics.median(r["secs"] for r in runs), 2)
                med_ram = round(statistics.median(r["peak_mb"] for r in runs))
                rec = {
                    "scale": f"{s}x",
                    "panel_rows": total_rows,
                    "parquet_MB": round(disk_mb, 1),
                    "method": method,
                    "median_seconds": med_s,
                    "peak_RAM_MB": med_ram,
                    "all_runs_s": [round(r["secs"], 2) for r in runs],
                }
                rows.append(rec)
                print(" ", rec)
                if method == "pandas" and med_ram > RAM_SAFE_LIMIT_MB:
                    skip_pandas = True
    finally:
        if SCALE_DIR.exists():
            shutil.rmtree(SCALE_DIR, ignore_errors=True)

    df_res = pd.DataFrame(rows)
    df_res.to_csv("outputs/part3_scaling.csv", index=False)
    print("\n=== SCALING SUMMARY ===")
    print(df_res.to_string(index=False))