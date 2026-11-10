import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import load_financials
import io
_buf = io.StringIO()
_orig_stdout = sys.stdout
sys.stdout = _buf

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

df = load_financials()
n = df.height
print(f"ROWS {n:,} | BANKS {df['CERT'].n_unique():,} | QUARTERS {df['REPDTE'].n_unique()}")

# 1. Missing values per column
miss = (df.null_count()
        .transpose(include_header=True, column_names=["nulls"])
        .with_columns((pl.col("nulls") / n * 100).round(2).alias("pct_missing"))
        .sort("nulls", descending=True))
print("\n[1] MISSING VALUES"); print(miss)
miss.write_csv(OUT / "part2_missing.csv")

# 2. Banks per quarter
per_q = df.group_by("REPDTE").agg(pl.len().alias("n_banks")).sort("REPDTE")
print("\n[2] BANKS PER QUARTER (first 3 / last 3 / min / max)")
print(per_q.head(3)); print(per_q.tail(3))
print("min", per_q["n_banks"].min(), "max", per_q["n_banks"].max())
per_q.write_csv(OUT / "part2_banks_per_quarter.csv")

# 3. Implausible values
print("\n[3] IMPLAUSIBLE VALUES")
print("ASSET <= 0      :", df.filter(pl.col("ASSET") <= 0).height)
print("ASSET is null   :", df["ASSET"].null_count())
print("EQ < 0 (negative equity):", df.filter(pl.col("EQ") < 0).height)
print("|ROA| > 50 (%)  :", df.filter(pl.col("ROA").abs() > 50).height)
print("|ROE| > 200 (%) :", df.filter(pl.col("ROE").abs() > 200).height)
print("DEP > ASSET     :", df.filter(pl.col("DEP") > pl.col("ASSET")).height)

# 4. Gaps in each bank's history and exits
d = (df.select("CERT", "REPDTE")
       .with_columns((pl.col("REPDTE").str.slice(0, 4).cast(pl.Int32) * 4
                      + (pl.col("REPDTE").str.slice(4, 2).cast(pl.Int32) - 1) // 3).alias("QIDX"))
       .sort("CERT", "QIDX")
       .with_columns(pl.col("QIDX").diff().over("CERT").alias("gap")))
gaps = d.filter(pl.col("gap") > 1)
print("\n[4] HISTORY GAPS")
print("rows preceded by a missing quarter:", gaps.height, "| banks affected:", gaps["CERT"].n_unique())
last = d.group_by("CERT").agg(pl.col("QIDX").max().alias("LAST_Q"))
max_q = d["QIDX"].max()
print("banks whose last report is before the final quarter (exited):", last.filter(pl.col("LAST_Q") < max_q).height)

# 5. Are income items cumulative within the year?
chk = (df.filter(pl.col("ASSET") > 0)
         .with_columns(pl.col("REPDTE").str.slice(4, 2).alias("MONTH"),
                       (pl.col("NETINC") / pl.col("ASSET") * 100).alias("NI_ASSET"))
         .group_by("MONTH").agg(pl.col("NI_ASSET").median().alias("median_NETINC_over_ASSET_pct"))
         .sort("MONTH"))
print("\n[5] NETINC / ASSET BY QUARTER-END MONTH (if ~ 1:2:3:4 => year-to-date cumulative)")
print(chk)
sys.stdout = _orig_stdout
text = _buf.getvalue()
Path("outputs/part2_quality_report.txt").write_text(text, encoding="utf-8")
print(text)