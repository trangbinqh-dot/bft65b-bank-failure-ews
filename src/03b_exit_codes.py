import polars as pl

pl.Config.set_tbl_rows(60)

f = pl.read_parquet("data_raw/fdic/failures.parquet")
i = pl.read_parquet("data_raw/fdic/institutions.parquet")
fin = pl.read_parquet("data_processed/financials_opt.parquet", columns=["CERT", "REPDTE"])

f = f.with_columns(
    pl.col("FAILDATE").str.to_date("%m/%d/%Y").alias("FD"),
    pl.col("CERT").cast(pl.Int32, strict=False),
)
i = i.with_columns(pl.col("CERT").cast(pl.Int32))
last = (fin.group_by("CERT")
           .agg(pl.col("REPDTE").max().alias("LAST_REPDTE"), pl.len().alias("N_QTRS"))
           .with_columns(pl.col("LAST_REPDTE").str.to_date("%Y%m%d").alias("LAST_DATE")))

print("[A] FAILURES: total", f.height, "| CERT null:", f["CERT"].null_count(),
      "| FAILDATE range:", f["FD"].min(), "->", f["FD"].max())

win = f.filter((pl.col("FD") >= pl.date(1984, 1, 1)) & pl.col("CERT").is_not_null())
win = win.join(last, on="CERT", how="left").with_columns(pl.col("LAST_DATE").is_not_null().alias("IN_PANEL"))
print("\n[B] FAILURES since 1984 with CERT, by type and presence in panel")
print(win.group_by("RESTYPE", "IN_PANEL").len().sort("RESTYPE", "IN_PANEL"))

fail = win.filter((pl.col("RESTYPE") == "FAILURE") & pl.col("IN_PANEL"))
print("\n[C] FAILURE (in panel) per year")
print(fail.group_by(pl.col("FD").dt.year().alias("YEAR")).len().sort("YEAR"))

print("\n[D] CHANGEC1 of banks that FAILED vs all other banks")
flag = fail.select("CERT").unique().with_columns(pl.lit(True).alias("FAILED"))
x = i.join(flag, on="CERT", how="left").with_columns(pl.col("FAILED").fill_null(False))
print(x.group_by("FAILED", "CHANGEC1").len().sort("FAILED", "len", descending=[True, True]).head(30))

print("\n[E] Days between last report and FAILDATE (failed banks)")
gap = fail.with_columns((pl.col("FD") - pl.col("LAST_DATE")).dt.total_days().alias("DAYS"))
print(gap.select("DAYS").describe())
print(gap.select(pl.col("DAYS").quantile(0.9).alias("p90")))