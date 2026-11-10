import polars as pl

df = pl.read_parquet("data_processed/financials_opt.parquet", columns=["CERT", "REPDTE", "NTLNLS", "NETINC"])
df = df.with_columns(pl.col("REPDTE").str.slice(0, 4).alias("Y"), pl.col("REPDTE").str.slice(4, 2).alias("M"))
for c in ["NTLNLS", "NETINC"]:
    w = (df.select("CERT", "Y", "M", c)
           .pivot(on="M", index=["CERT", "Y"], values=c)
           .drop_nulls())
    mono = w.filter((pl.col("06") >= pl.col("03")) & (pl.col("09") >= pl.col("06")) & (pl.col("12") >= pl.col("09")))
    # use only bank-years where the quantity is positive in December, to avoid noise from zeros
    pos = w.filter(pl.col("12") > 0)
    mono_pos = pos.filter((pl.col("06") >= pl.col("03")) & (pl.col("09") >= pl.col("06")) & (pl.col("12") >= pl.col("09")))
    print(c, "| bank-years:", w.height, "| monotone share:", round(mono.height / w.height, 3),
          "| among Dec>0:", round(mono_pos.height / pos.height, 3))