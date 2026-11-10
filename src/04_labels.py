import sys
from datetime import date
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import load_financials

pl.Config.set_tbl_rows(60)
Path("data_processed").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)

LAST_Q = date(2026, 6, 30)       # last quarter downloaded
LABEL_END = date(2025, 6, 30)    # labels complete up to here

df = load_financials().with_columns(pl.col("REPDTE").str.to_date("%Y%m%d").alias("QDATE"))

fail = (pl.read_parquet("data_raw/fdic/failures.parquet")
        .with_columns(pl.col("FAILDATE").str.to_date("%m/%d/%Y").alias("FD"),
                      pl.col("CERT").cast(pl.Int32, strict=False))
        .filter((pl.col("RESTYPE") == "FAILURE") & pl.col("CERT").is_not_null()))
print("failure rows:", fail.height, "| distinct CERT:", fail["CERT"].n_unique())
fail = fail.group_by("CERT").agg(pl.col("FD").min().alias("FD"), pl.col("COST").first().alias("COST"))

p = df.join(fail, on="CERT", how="left")

# rows reported on/after the failure date: not a live bank any more
post = p.filter(pl.col("FD").is_not_null() & (pl.col("QDATE") >= pl.col("FD")))
print("rows on/after failure date (dropped):", post.height, "| banks:", post["CERT"].n_unique())
p = p.filter(pl.col("FD").is_null() | (pl.col("QDATE") < pl.col("FD")))

p = p.with_columns(
    (pl.col("FD").is_not_null()
     & (pl.col("FD") > pl.col("QDATE"))
     & (pl.col("FD") <= pl.col("QDATE").dt.offset_by("12mo"))).cast(pl.Int8).alias("y")
)

# exit type per bank
last = p.group_by("CERT").agg(pl.col("QDATE").max().alias("LAST"))
ex = (last.join(fail.select("CERT", "FD"), on="CERT", how="left")
      .with_columns(pl.when(pl.col("FD").is_not_null()).then(pl.lit("failure"))
                    .when(pl.col("LAST") >= LAST_Q).then(pl.lit("active at end"))
                    .otherwise(pl.lit("other exit (merger etc.)")).alias("EXIT_TYPE")))
print("\n[EXIT TYPES]"); print(ex.group_by("EXIT_TYPE").len().sort("len", descending=True))
ex.select("CERT", "EXIT_TYPE").write_parquet("data_processed/exit_types.parquet")

lab = p.filter(pl.col("QDATE") <= LABEL_END)
print("\nlabelled rows:", lab.height, "| positives:", lab["y"].sum(),
      f"| rate {lab['y'].mean() * 100:.3f}%")

cands = [("A", date(2006, 12, 31), date(2008, 3, 31)),
         ("B", date(2009, 12, 31), date(2011, 3, 31)),
         ("C", date(2010, 12, 31), date(2012, 3, 31))]
print("\n[SPLIT CANDIDATES] (gap = 4 quarters)")
for name, de, ts in cands:
    dev = lab.filter(pl.col("QDATE") <= de)
    te = lab.filter(pl.col("QDATE") >= ts)
    f = lambda d: d.filter(pl.col("y") == 1)["CERT"].n_unique()
    print(f"{name}: dev<= {de}: rows {dev.height:,}, pos rows {dev['y'].sum():,}, failed banks {f(dev)}"
          f" | test>= {ts}: rows {te.height:,}, pos rows {te['y'].sum():,}, failed banks {f(te)}")

lab.drop("FD").write_parquet("data_processed/panel_labeled.parquet", compression="zstd")
print("saved data_processed/panel_labeled.parquet")