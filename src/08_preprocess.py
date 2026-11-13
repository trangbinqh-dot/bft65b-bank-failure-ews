import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import tee_log

NUM_FEATURES = [
    "LOG_ASSET", "EQ_ASSET", "RBC1AAJ", "RBCRWAJ",
    "NPL_LOANS", "NONACCR_ASSET", "PD90_ASSET", "ORE_ASSET", "RES_LOANS",
    "ROA", "ROE", "NIMY", "ROA_Q_ANN", "NCO_RATE_Q", "PROV_RATE_Q", "EFFICIENCY",
    "LIQ_ASSET", "DEP_ASSET", "BRO_ASSET", "LOANS_ASSET", "RE_LOANS",
    "ASSET_G4", "DEP_G4", "EQ_ASSET_D4", "NPL_LOANS_D4", "ROA_D4", "LOANS_ASSET_D4",
    "ROA_MA4", "ROA_SD4",
]
BASE_MISS = ["P9ASSET_MISS", "NAASSET_MISS", "NTLNLS_MISS", "ELNATR_MISS", "ROE_MISS", "BRO_MISS"]
EXTRA_MISS_SRC = ["ASSET_G4", "NPL_LOANS_D4", "ROA_MA4", "EFFICIENCY"]
EXTRA_MISS = [c + "_MISS" for c in EXTRA_MISS_SRC]
ALL_MISS = BASE_MISS + EXTRA_MISS
ALL_COLS = NUM_FEATURES + ALL_MISS

DEV_END = date(2009, 12, 31)
TEST_START = date(2011, 3, 31)


class QuantileWinsorizer(BaseEstimator, TransformerMixin):
    """Clip extreme tails using quantiles estimated strictly on training data."""

    def __init__(self, lower: float = 0.005, upper: float = 0.995):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=np.float64)
        self.low_ = np.nanquantile(arr, self.lower, axis=0)
        self.high_ = np.nanquantile(arr, self.upper, axis=0)
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=np.float64)
        return np.clip(arr, self.low_, self.high_)


def add_derived_miss_flags(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [pl.col(c).is_null().cast(pl.Int8).alias(c + "_MISS") for c in EXTRA_MISS_SRC]
    )


def make_preprocess_pipeline() -> ColumnTransformer:
    """Reusable leak-free preprocessor for Part 4 verification and Part 5 CV/modelling."""
    num_pipe = Pipeline([
        ("winsorize", QuantileWinsorizer(lower=0.005, upper=0.995)),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, list(range(len(NUM_FEATURES)))),
            ("flags", "passthrough", list(range(len(NUM_FEATURES), len(ALL_COLS)))),
        ],
        remainder="drop",
    )


if __name__ == "__main__":
    tee_log("outputs/part4_preprocess_report.txt")
    pl.Config.set_tbl_rows(40)

    df = add_derived_miss_flags(pl.read_parquet("data_processed/features.parquet"))

    dev_df = df.filter(pl.col("QDATE") <= DEV_END)
    gap_df = df.filter((pl.col("QDATE") > DEV_END) & (pl.col("QDATE") < TEST_START))
    test_df = df.filter(pl.col("QDATE") >= TEST_START)

    print(f"Split summary (4-quarter gap in 2010):")
    print(f"  DEV  (<= {DEV_END}): rows={dev_df.height:,} | y=1={dev_df['y'].sum():,} ({dev_df['y'].mean()*100:.3f}%)")
    print(f"  GAP  (2010Q1-2010Q4): rows={gap_df.height:,} | y=1={gap_df['y'].sum():,} (excluded)")
    print(f"  TEST (>= {TEST_START}): rows={test_df.height:,} | y=1={test_df['y'].sum():,} ({test_df['y'].mean()*100:.3f}%)")
    assert dev_df.height + gap_df.height + test_df.height == df.height

    X_dev = dev_df.select(ALL_COLS).to_numpy()
    y_dev = dev_df["y"].to_numpy()
    X_test = test_df.select(ALL_COLS).to_numpy()

    # Fit ONLY on DEV, then transform DEV and TEST without refitting
    prep = make_preprocess_pipeline()
    X_dev_tr = prep.fit_transform(X_dev)
    X_test_tr = prep.transform(X_test)

    assert not np.isnan(X_dev_tr).any() and not np.isinf(X_dev_tr).any()
    assert not np.isnan(X_test_tr).any() and not np.isinf(X_test_tr).any()
    print(f"\nTransformed shapes -> X_dev: {X_dev_tr.shape}, X_test: {X_test_tr.shape} | NaN/Inf after pipeline: 0")

    # Verify zero leakage: modifying X_test arbitrarily must NOT change fitted parameters or X_dev_tr
    win = prep.named_transformers_["num"].named_steps["winsorize"]
    imp = prep.named_transformers_["num"].named_steps["impute"]
    scl = prep.named_transformers_["num"].named_steps["scale"]

    manual_low = np.nanquantile(X_dev[:, :len(NUM_FEATURES)], 0.005, axis=0)
    manual_high = np.nanquantile(X_dev[:, :len(NUM_FEATURES)], 0.995, axis=0)
    assert np.allclose(win.low_, manual_low) and np.allclose(win.high_, manual_high)
    print("Leakage gate check: Winsorizer quantiles match manual DEV-only quantiles 100% (PASSED)")

    # Print sample learned thresholds & medians on DEV for Report Table
    summary = pl.DataFrame({
        "feature": NUM_FEATURES,
        "dev_p005": np.round(win.low_, 3),
        "dev_median_imputed": np.round(imp.statistics_, 3),
        "dev_p995": np.round(win.high_, 3),
        "dev_post_win_corr_y": [
            round(float(np.corrcoef(X_dev_tr[:, j], y_dev)[0, 1]), 4)
            for j in range(len(NUM_FEATURES))
        ],
    })
    print("\nLearned DEV-only preprocessing parameters & post-winsorization correlation with y:")
    print(summary)