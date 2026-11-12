import sys
import time
from datetime import date
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import tee_log

tee_log("outputs/part3_polars_report.txt")
pl.Config.set_tbl_rows(20)
pl.Config.set_tbl_cols(12)
pl.Config.set_fmt_str_lengths(60)

PANEL = "data_processed/panel_clean.parquet"
DEV_END = date(2009, 12, 31)


def build(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Core transformations; every window looks backwards only."""
    lf = (
        lf.filter(pl.col("QDATE") <= DEV_END)
        .select("CERT", "QDATE", "ASSET", "EQ", "LNLSGR", "LNATRES", "NCLNLS", "ROA", "y")
        .with_columns(
            (pl.col("EQ") / pl.col("ASSET") * 100).alias("EQ_RATIO"),
            (pl.col("NCLNLS") / pl.col("LNLSGR") * 100).alias("NPL_RATIO"),
            (pl.col("LNATRES") / pl.col("LNLSGR") * 100).alias("RES_RATIO"),
        )
        .sort("CERT", "QDATE")
        .with_columns(
            (pl.col("ASSET") / pl.col("ASSET").shift(4).over("CERT") - 1).alias("ASSET_G4"),
            (pl.col("EQ_RATIO") - pl.col("EQ_RATIO").shift(4).over("CERT")).alias("EQ_RATIO_D4"),
            pl.col("ROA").rolling_mean(window_size=4).over("CERT").alias("ROA_MA4"),
        )
    )
    return lf


lf = build(pl.scan_parquet(PANEL))
q = (lf.group_by("QDATE")
       .agg(pl.len().alias("n_rows"),
            pl.col("EQ_RATIO").median().alias("med_EQ_RATIO"),
            pl.col("NPL_RATIO").median().alias("med_NPL_RATIO"),
            pl.col("ASSET_G4").median().alias("med_ASSET_G4"),
            pl.col("ROA_MA4").median().alias("med_ROA_MA4"),
            pl.col("y").sum().alias("n_positive"))
       .sort("QDATE"))

plan_raw = q.explain(optimized=False)
plan_opt = q.explain()
Path("outputs/part3_plan_unoptimized.txt").write_text(plan_raw, encoding="utf-8")
Path("outputs/part3_plan_optimized.txt").write_text(plan_opt, encoding="utf-8")
print("=== UNOPTIMIZED PLAN ===\n" + plan_raw)
print("\n=== OPTIMIZED PLAN ===\n" + plan_opt)

# run lazy (optimised) and eager (read everything first) for a first timing
t0 = time.perf_counter(); res_lazy = q.collect(); t_lazy = time.perf_counter() - t0

t0 = time.perf_counter()
res_eager = build(pl.read_parquet(PANEL).lazy()).group_by("QDATE").agg(
    pl.len().alias("n_rows"),
    pl.col("EQ_RATIO").median().alias("med_EQ_RATIO"),
    pl.col("NPL_RATIO").median().alias("med_NPL_RATIO"),
    pl.col("ASSET_G4").median().alias("med_ASSET_G4"),
    pl.col("ROA_MA4").median().alias("med_ROA_MA4"),
    pl.col("y").sum().alias("n_positive")).sort("QDATE").collect(optimizations=pl.QueryOptFlags.none())
t_eager = time.perf_counter() - t0

print(f"\nlazy optimised: {t_lazy:.3f}s | no optimisation (reads all columns, no pushdown): {t_eager:.3f}s")
print("same result:", res_lazy.equals(res_eager))
print(res_lazy.head(8))
print(res_lazy.tail(4))
print("rows:", res_lazy["n_rows"].sum(), "| positives:", res_lazy["n_positive"].sum())