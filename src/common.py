from pathlib import Path
import polars as pl

KEY_COLS = ("CERT", "REPDTE", "STNAME")


def load_financials(folder: str = "data_raw/fdic/financials") -> pl.DataFrame:
    """Read all quarterly Parquet files and harmonise the schema before concatenation."""
    frames = []
    for f in sorted(Path(folder).glob("*.parquet")):
        df = pl.read_parquet(f)
        df = df.with_columns(
            [pl.col(c).cast(pl.Float64) for c in df.columns if c not in KEY_COLS]
        ).with_columns(
            pl.col("CERT").cast(pl.Int32),
            pl.col("REPDTE").cast(pl.String),
        )
        frames.append(df)
    return pl.concat(frames, how="diagonal_relaxed")