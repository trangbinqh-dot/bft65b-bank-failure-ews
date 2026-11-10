import polars as pl

pl.Config.set_tbl_rows(40)
df = pl.read_parquet("data_processed/financials_opt.parquet")
df = df.filter(pl.col("ASSET") > 0).with_columns(
    pl.col("REPDTE").str.slice(4, 2).alias("M"))

cols = ["NETINC", "INTINC", "EINTEXP", "NONII", "NONIX", "ELNATR", "NTLNLS",
        "ROA", "ROE", "NIMY"]
rows = []
for c in cols:
    r = (df.group_by("M").agg((pl.col(c) / pl.col("ASSET") * 100).median().alias("v"))
           .sort("M"))
    v = r["v"].to_list()
    rows.append({"col": c, "Mar": v[0], "Jun": v[1], "Sep": v[2], "Dec": v[3],
                 "Dec/Mar": v[3] / v[0] if v[0] else None})
print(pl.DataFrame(rows))