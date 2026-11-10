from pathlib import Path

import polars as pl

pl.Config.set_tbl_rows(40)
YTD = ["NETINC", "INTINC", "EINTEXP", "NONII", "NONIX", "ELNATR", "NTLNLS"]

p = pl.read_parquet("data_processed/panel_labeled.parquet").sort("CERT", "QDATE")
p = p.with_columns(
    pl.col("QDATE").dt.year().alias("YEAR"),
    pl.col("QDATE").dt.month().alias("MONTH"),
    (pl.col("QDATE").dt.year() * 4 + (pl.col("QDATE").dt.month() - 1) // 3).alias("QIDX"),
)

# previous row of the same bank; usable only if it is the previous quarter of the same year
exprs = []
for c in YTD:
    prev = pl.col(c).shift(1).over("CERT")
    prev_q = pl.col("QIDX").shift(1).over("CERT")
    same_year_prev = (prev_q == pl.col("QIDX") - 1) & (pl.col("MONTH") != 3)
    exprs.append(
        pl.when(pl.col("MONTH") == 3).then(pl.col(c))
          .when(same_year_prev).then(pl.col(c) - prev)
          .otherwise(None)          # missing previous quarter -> NaN, not guessed
          .alias(c + "_Q")
    )
p = p.with_columns(exprs)

# missing flags (information only; no imputation is fitted here)
flag_cols = ["P3ASSET", "P9ASSET", "NAASSET", "NTLNLS", "ELNATR", "ROE", "BRO"]
p = p.with_columns([pl.col(c).is_null().cast(pl.Int8).alias(c + "_MISS") for c in flag_cols])

print("rows:", p.height, "| banks:", p["CERT"].n_unique(), "| positives:", p["y"].sum())
print("null share of quarterly columns:")
for c in YTD:
    print(f"  {c}_Q: {p[c + '_Q'].null_count() / p.height * 100:.2f}%")

# sanity check: Q2 of year = YTD(Jun) - YTD(Mar) must equal NETINC_Q
chk = p.filter((pl.col("MONTH") == 3)).select((pl.col("NETINC") - pl.col("NETINC_Q")).abs().sum())
print("Q1 check (should be 0):", chk.item())

p.write_parquet("data_processed/panel_clean.parquet", compression="zstd")
print("saved data_processed/panel_clean.parquet")