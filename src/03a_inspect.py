import polars as pl

pl.Config.set_tbl_rows(30)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(40)

f = pl.read_parquet("data_raw/fdic/failures.parquet")
i = pl.read_parquet("data_raw/fdic/institutions.parquet")

print("=== FAILURES ===")
print(f.schema)
print(f.head(5))
print(f["RESTYPE"].value_counts())
print(f.select("COST").describe())
print("=== INSTITUTIONS ===")
print(i.schema)
print(i.head(5))
print(i["ACTIVE"].value_counts())
print(i["CHANGEC1"].value_counts().sort("count", descending=True).head(15))