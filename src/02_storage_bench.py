import shutil
import statistics
import sys
import time
from pathlib import Path

import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import load_financials

BENCH = Path("data_processed/bench")
OUT = Path("outputs")
BENCH.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
REPEAT = 3
COLS = ["CERT", "REPDTE", "ASSET", "EQ", "ROA"]
RATIO_COLS = ["ROA", "ROE", "NIMY", "RBC1AAJ", "RBCRWAJ"]


def size_mb(p: Path) -> float:
    if p.is_file():
        return p.stat().st_size / 2**20
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 2**20


def median_time(fn) -> float:
    ts = []
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


df = load_financials()
n_dup = df.height - df.unique(subset=["CERT", "REPDTE"]).height
print(f"rows={df.height:,} banks={df['CERT'].n_unique():,} duplicate keys={n_dup}")

# Optimised schema: money stays Float64; only percentage ratios -> Float32
df_opt = df.with_columns(
    [pl.col(c).cast(pl.Float32) for c in RATIO_COLS if c in df.columns]
    + [pl.col("STNAME").cast(pl.Categorical)]
)
df_year = df_opt.with_columns(pl.col("REPDTE").str.slice(0, 4).alias("YEAR"))

rows = []


def bench(name, path, writer, full_read, selective):
    if path.exists():
        shutil.rmtree(path) if path.is_dir() else path.unlink()
    t0 = time.perf_counter()
    writer(path)
    w = time.perf_counter() - t0
    rows.append({
        "Layout": name,
        "Size (MB)": round(size_mb(path), 1),
        "Write (s)": round(w, 2),
        "Full read (s)": round(median_time(lambda: full_read(path)), 2),
        "5 cols + year>=2008 (s)": round(median_time(lambda: selective(path)), 3),
    })
    print(rows[-1])


flt = lambda lf: lf.select(COLS).filter(pl.col("REPDTE").cast(pl.String) >= "20080331").collect()

bench("CSV", BENCH / "fin.csv", lambda p: df.write_csv(p),
      lambda p: pl.read_csv(p), lambda p: flt(pl.scan_csv(p)))

for comp in ("uncompressed", "snappy", "zstd"):
    bench(f"Parquet single ({comp})", BENCH / f"fin_{comp}.parquet",
          lambda p, c=comp: df.write_parquet(p, compression=c),
          lambda p: pl.read_parquet(p), lambda p: flt(pl.scan_parquet(p)))

bench("Parquet single, optimised schema (zstd)", BENCH / "fin_opt.parquet",
      lambda p: df_opt.write_parquet(p, compression="zstd"),
      lambda p: pl.read_parquet(p), lambda p: flt(pl.scan_parquet(p)))

bench("Parquet hive-partitioned by YEAR (zstd)", BENCH / "fin_year",
      lambda p: df_year.write_parquet(p, compression="zstd", partition_by="YEAR"),
      lambda p: pl.read_parquet(p / "**/*.parquet"),
      lambda p: pl.scan_parquet(p / "**/*.parquet", hive_partitioning=True)
      .filter(pl.col("YEAR") >= "2008").select(COLS).collect())

res = pd.DataFrame(rows)
csv_size = res.loc[0, "Size (MB)"]
res["Compression vs CSV (x)"] = (csv_size / res["Size (MB)"]).round(1)
print("\n", res.to_string(index=False))
res.to_csv(OUT / "part1_storage_benchmark.csv", index=False)

# Save the best layout for later parts
df_opt.write_parquet("data_processed/financials_opt.parquet", compression="zstd")
print("Saved data_processed/financials_opt.parquet")