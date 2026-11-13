import sys
from datetime import date
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import tee_log

tee_log("outputs/part4_features_report.txt")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)

DEV_END = date(2009, 12, 31)


def safe_div(a, b):
    return a / pl.when(b != 0).then(b)     # zero denominator -> null (no inf / NaN)


def compute_features(p: pl.DataFrame) -> pl.DataFrame:
    # ---------- 1. same-quarter ratios (only information reported at quarter q) ----------
    net_op_rev_q = (pl.col("INTINC_Q") - pl.col("EINTEXP_Q")) + pl.col("NONII_Q")
    p = p.with_columns(
        pl.col("ASSET").log().alias("LOG_ASSET"),                                   # size
        safe_div(pl.col("EQ"), pl.col("ASSET")).mul(100).alias("EQ_ASSET"),         # capital
        safe_div(pl.col("NCLNLS"), pl.col("LNLSGR")).mul(100).alias("NPL_LOANS"),   # asset quality
        safe_div(pl.col("NAASSET"), pl.col("ASSET")).mul(100).alias("NONACCR_ASSET"),
        safe_div(pl.col("P9ASSET"), pl.col("ASSET")).mul(100).alias("PD90_ASSET"),
        safe_div(pl.col("ORE"), pl.col("ASSET")).mul(100).alias("ORE_ASSET"),
        safe_div(pl.col("LNATRES"), pl.col("LNLSGR")).mul(100).alias("RES_LOANS"),  # reserves
        safe_div(pl.col("LNLSGR"), pl.col("ASSET")).mul(100).alias("LOANS_ASSET"),  # loans / assets
        safe_div(pl.col("DEP"), pl.col("ASSET")).mul(100).alias("DEP_ASSET"),       # funding
        safe_div(pl.col("BRO"), pl.col("ASSET")).mul(100).alias("BRO_ASSET"),       # brokered deposits
        safe_div(pl.col("CHBAL") + pl.col("SC"), pl.col("ASSET")).mul(100).alias("LIQ_ASSET"),  # liquidity
        safe_div(pl.col("LNRE"), pl.col("LNLSGR")).mul(100).alias("RE_LOANS"),      # concentration
        safe_div(pl.col("NETINC_Q"), pl.col("ASSET")).mul(400).alias("ROA_Q_ANN"),  # earnings (quarterly, annualised)
        safe_div(pl.col("NTLNLS_Q"), pl.col("LNLSGR")).mul(400).alias("NCO_RATE_Q"),
        safe_div(pl.col("ELNATR_Q"), pl.col("LNLSGR")).mul(400).alias("PROV_RATE_Q"),
        (pl.col("NONIX_Q") / pl.when(net_op_rev_q > 0).then(net_op_rev_q)).mul(100).alias("EFFICIENCY"),
    )

    # ---------- 2. lags by TRUE quarter index (join to the same bank 4 quarters earlier) ----------
    BASE = ["ASSET", "DEP", "EQ_ASSET", "NPL_LOANS", "ROA", "LOANS_ASSET"]
    past = p.select(["CERT", (pl.col("QIDX") + 4).alias("QIDX")] + [pl.col(c).alias(c + "_L4") for c in BASE])
    p = p.join(past, on=["CERT", "QIDX"], how="left")

    p = p.with_columns(
        (safe_div(pl.col("ASSET"), pl.col("ASSET_L4")) - 1).mul(100).alias("ASSET_G4"),
        (safe_div(pl.col("DEP"), pl.col("DEP_L4")) - 1).mul(100).alias("DEP_G4"),
        (pl.col("EQ_ASSET") - pl.col("EQ_ASSET_L4")).alias("EQ_ASSET_D4"),
        (pl.col("NPL_LOANS") - pl.col("NPL_LOANS_L4")).alias("NPL_LOANS_D4"),
        (pl.col("ROA") - pl.col("ROA_L4")).alias("ROA_D4"),
        (pl.col("LOANS_ASSET") - pl.col("LOANS_ASSET_L4")).alias("LOANS_ASSET_D4"),
    ).drop([c + "_L4" for c in BASE])

    # ---------- 3. backward-looking rolling stats (current + previous 3 consecutive quarters) ----------
    p = p.sort("CERT", "QIDX").with_columns(
        pl.col("ROA").rolling_mean(window_size=4, min_samples=4).over("CERT").alias("ROA_MA4"),
        pl.col("ROA").rolling_std(window_size=4, min_samples=4).over("CERT").alias("ROA_SD4"),
    )
    span = pl.col("QIDX") - pl.col("QIDX").shift(3).over("CERT")
    p = p.with_columns(
        pl.when(span == 3).then(pl.col("ROA_MA4")).alias("ROA_MA4"),
        pl.when(span == 3).then(pl.col("ROA_SD4")).alias("ROA_SD4"),
    )
    return p


raw_panel = pl.read_parquet("data_processed/panel_clean.parquet")
p = compute_features(raw_panel)

FEATURES = [
    "LOG_ASSET", "EQ_ASSET", "RBC1AAJ", "RBCRWAJ",
    "NPL_LOANS", "NONACCR_ASSET", "PD90_ASSET", "ORE_ASSET", "RES_LOANS",
    "ROA", "ROE", "NIMY", "ROA_Q_ANN", "NCO_RATE_Q", "PROV_RATE_Q", "EFFICIENCY",
    "LIQ_ASSET", "DEP_ASSET", "BRO_ASSET", "LOANS_ASSET", "RE_LOANS",
    "ASSET_G4", "DEP_G4", "EQ_ASSET_D4", "NPL_LOANS_D4", "ROA_D4", "LOANS_ASSET_D4",
    "ROA_MA4", "ROA_SD4",
]
MISS = [c for c in p.columns if c.endswith("_MISS")]
KEEP = ["CERT", "QDATE", "QIDX", "y"] + FEATURES + MISS
out = p.select(
    ["CERT", "QDATE", "QIDX", "y"]
    + [pl.col(c).cast(pl.Float64) for c in FEATURES]
    + MISS
).sort("CERT", "QIDX")

# ---------- 4. strict point-in-time & leakage checks ----------
print("features:", len(FEATURES), "| missing flags:", MISS)
forbidden = {"COST", "FD", "FAILDATE", "P3ASSET", "P3ASSET_MISS"}
assert not (forbidden & set(out.columns)), "forbidden column present"
assert out.height == 1_663_876, out.height
assert out["y"].sum() == 10_860

# Check no +/-inf or IEEE NaN leaked into any feature column
n_bad_num = sum(
    out.select((pl.col(c).is_infinite() | pl.col(c).is_nan()).sum()).item()
    for c in FEATURES
)
assert n_bad_num == 0, f"Found {n_bad_num} inf/NaN values!"
print("rows:", out.height, "| positives:", out["y"].sum(), "| inf/NaN count:", n_bad_num)

# Point-in-time truncation test: features at cutoff_date computed WITHOUT future rows must match 100%
cutoff = date(2005, 12, 31)
trunc_out = (
    compute_features(raw_panel.filter(pl.col("QDATE") <= cutoff))
    .filter(pl.col("QDATE") == cutoff)
    .select(["CERT"] + [pl.col(c).cast(pl.Float64) for c in FEATURES])
    .sort("CERT")
)
full_at_cutoff = (
    out.filter(pl.col("QDATE") == cutoff)
    .select(["CERT"] + FEATURES)
    .sort("CERT")
)
assert trunc_out.equals(full_at_cutoff, null_equal=True), "Look-ahead bias detected in feature computation!"
print(f"Point-in-time truncation check at {cutoff} ({trunc_out.height:,} banks): PASSED 100% EXACT MATCH")

# Inspect correlations & missingness ONLY on Development period (<= 2009-12-31) to never touch Test set
dev = out.filter(pl.col("QDATE") <= DEV_END)
print(f"\n=== DIAGNOSTICS ON DEV PERIOD ONLY (<= {DEV_END}, rows={dev.height:,}, y=1={dev['y'].sum():,}) ===")

corr = (
    dev.select([pl.corr(c, "y").round(4).alias(c) for c in FEATURES])
    .transpose(include_header=True, column_names=["corr_with_y_dev"])
)
print("\nCorrelation with y on DEV (real EWS features typically |r| < 0.25):")
print(corr.sort(pl.col("corr_with_y_dev").abs(), descending=True))

print("\nNull share per feature on DEV (%):")
print(
    dev.select([(pl.col(c).null_count() / dev.height * 100).round(2).alias(c) for c in FEATURES])
    .transpose(include_header=True, column_names=["null_pct_dev"])
    .sort("null_pct_dev", descending=True)
)

out.write_parquet("data_processed/features.parquet", compression="zstd")
print("\nsaved data_processed/features.parquet")