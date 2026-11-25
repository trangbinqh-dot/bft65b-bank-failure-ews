"""Part 5 supplementary Dev-only checks (strictly QDATE <= 2009-12-31; Test untouched).

1. Pre-1990 Basel I zero-coding sensitivity on 3-fold expanding-window CV (HistGBDT):
   Replaces pre-1990 zeros in RBC1AAJ and RBCRWAJ with NaN in memory (adding two
   in-memory missingness indicators RBC1AAJ_MISS and RBCRWAJ_MISS) without modifying
   data_processed/features.parquet or any locked model/threshold.
2. Out-of-fold probability calibration (reliability) table on Dev OOF predictions:
   Quantifies probability over/under-confidence across predicted probability bins
   to support the dynamic recalibration recommendation in Section 7.2.
"""
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Import exact custom QuantileWinsorizer and feature lists from 08_preprocess.py
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prep08", ROOT / "src" / "08_preprocess.py")
prep08 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep08)

QuantileWinsorizer = prep08.QuantileWinsorizer
CONT_COLS = list(prep08.CONT_COLS)
MISS_COLS = list(prep08.MISS_COLS)
SEED = 42

FOLDS = [
    ("Fold1_SL_1990_1993", date(1988, 12, 31), date(1990, 3, 31), date(1993, 12, 31)),
    ("Fold2_Calm_2000_2006", date(1998, 12, 31), date(2000, 3, 31), date(2006, 12, 31)),
    ("Fold3_GFC_2008_2009", date(2006, 12, 31), date(2008, 3, 31), date(2009, 12, 31)),
]


def build_hgb_pipe(cont_cols: list[str], miss_cols: list[str]) -> Pipeline:
    num_pipe = Pipeline([
        ("winsor", QuantileWinsorizer(lower=0.005, upper=0.995)),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, cont_cols),
            ("flags", "passthrough", miss_cols),
        ],
        remainder="drop",
    )
    clf = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=100,
        l2_regularization=1.0,
        random_state=SEED,
    )
    return Pipeline([("prep", pre), ("clf", clf)])


def main() -> None:
    feat_path = ROOT / "data_processed" / "features.parquet"
    # Strictly load Development period only (<= 2009-12-31); Test set is NEVER loaded
    df_dev = (
        pl.scan_parquet(feat_path)
        .filter(pl.col("QDATE") <= date(2009, 12, 31))
        .collect()
    )
    assert df_dev["QDATE"].max() <= date(2009, 12, 31), "Leakage guard failed!"
    print(f"Loaded Dev-only panel: {df_dev.height:,} rows (max QDATE = {df_dev['QDATE'].max()})")

    # Convert to pandas once for sklearn Pipeline
    pdf_base = df_dev.select(["QDATE", "y"] + CONT_COLS + MISS_COLS).to_pandas()

    # Construct in-memory variant: pre-1990 zeros in RBC1AAJ / RBCRWAJ -> NaN + *_MISS flags
    pdf_alt = pdf_base.copy()
    pre1990 = pd.to_datetime(pdf_alt["QDATE"]).dt.year < 1990
    for col in ["RBC1AAJ", "RBCRWAJ"]:
        mask_zero = pre1990 & (pdf_alt[col] == 0.0)
        n_replaced = int(mask_zero.sum())
        pdf_alt.loc[mask_zero, col] = np.nan
        pdf_alt[f"{col}_MISS"] = pdf_alt[col].isna().astype(np.float64)
        print(f"In-memory pre-1990 {col} == 0 -> NaN: {n_replaced:,} rows")

    alt_miss_cols = MISS_COLS + ["RBC1AAJ_MISS", "RBCRWAJ_MISS"]

    rows_comp = []
    oof_y, oof_p_base, oof_fold = [], [], []

    for fname, tr_end, va_start, va_end in FOLDS:
        assert (va_start - tr_end).days > 400, f"Purge gap < 4 quarters on {fname}"
        qdates = pd.to_datetime(pdf_base["QDATE"]).dt.date
        tr_mask = qdates <= tr_end
        va_mask = (qdates >= va_start) & (qdates <= va_end)

        y_tr = pdf_base.loc[tr_mask, "y"].to_numpy()
        y_va = pdf_base.loc[va_mask, "y"].to_numpy()

        # 1. Baseline HistGBDT (exact locked specification)
        pipe_base = build_hgb_pipe(CONT_COLS, MISS_COLS)
        pipe_base.fit(pdf_base.loc[tr_mask, CONT_COLS + MISS_COLS], y_tr)
        p_base = pipe_base.predict_proba(pdf_base.loc[va_mask, CONT_COLS + MISS_COLS])[:, 1]

        # 2. Pre-1990 RBC zero -> NaN sensitivity variant
        pipe_alt = build_hgb_pipe(CONT_COLS, alt_miss_cols)
        pipe_alt.fit(pdf_alt.loc[tr_mask, CONT_COLS + alt_miss_cols], y_tr)
        p_alt = pipe_alt.predict_proba(pdf_alt.loc[va_mask, CONT_COLS + alt_miss_cols])[:, 1]

        pr_base = average_precision_score(y_va, p_base)
        pr_alt = average_precision_score(y_va, p_alt)
        roc_base = roc_auc_score(y_va, p_base)
        roc_alt = roc_auc_score(y_va, p_alt)

        rows_comp.append({
            "fold": fname,
            "n_val": int(va_mask.sum()),
            "n_pos": int(y_va.sum()),
            "pr_auc_baseline": round(pr_base, 4),
            "pr_auc_rbc_null": round(pr_alt, 4),
            "pr_auc_diff": round(pr_alt - pr_base, 4),
            "roc_auc_baseline": round(roc_base, 4),
            "roc_auc_rbc_null": round(roc_alt, 4),
        })

        oof_y.append(y_va)
        oof_p_base.append(p_base)
        oof_fold.extend([fname] * len(y_va))

    df_comp = pd.DataFrame(rows_comp)
    mean_base = df_comp["pr_auc_baseline"].mean()
    mean_alt = df_comp["pr_auc_rbc_null"].mean()

    # OOF reliability / calibration table across probability bins (baseline HistGBDT)
    y_all = np.concatenate(oof_y)
    p_all = np.concatenate(oof_p_base)
    bins = [0.0, 0.01, 0.05, 0.15, 0.30, 0.50, 1.0001]
    labels = ["[0.00, 0.01)", "[0.01, 0.05)", "[0.05, 0.15)", "[0.15, 0.30)", "[0.30, 0.50)", "[0.50, 1.00]"]
    b_idx = pd.cut(p_all, bins=bins, labels=labels, right=False)
    cal_df = (
        pd.DataFrame({"bin": b_idx, "y": y_all, "p": p_all})
        .groupby("bin", observed=False)
        .agg(
            n_obs=("y", "count"),
            n_pos=("y", "sum"),
            mean_pred_prob=("p", "mean"),
            actual_fail_rate=("y", "mean"),
        )
        .reset_index()
    )
    cal_df["pred_to_actual_ratio"] = (cal_df["mean_pred_prob"] / cal_df["actual_fail_rate"].replace(0, np.nan)).round(2)

    out_txt = ROOT / "outputs" / "part5_dev_sensitivity_and_calibration.txt"
    out_csv1 = ROOT / "outputs" / "part5_rbc_pre1990_sensitivity.csv"
    out_csv2 = ROOT / "outputs" / "part5_dev_oof_calibration.csv"
    df_comp.to_csv(out_csv1, index=False)
    cal_df.to_csv(out_csv2, index=False)

    lines = [
        "=== SUPPLEMENTARY DEV-ONLY CHECKS (QDATE <= 2009-12-31; TEST UNTOUCHED) ===",
        f"Dev rows loaded: {df_dev.height:,} | Max QDATE: {df_dev['QDATE'].max()}",
        "",
        "[1] PRE-1990 RBC ZERO-TO-NULL SENSITIVITY (HistGBDT 3-Fold Expanding CV)",
        df_comp.to_string(index=False),
        f"Mean 3-fold PR-AUC: baseline = {mean_base:.4f} | pre-1990 RBC 0->NaN = {mean_alt:.4f} | diff = {mean_alt - mean_base:+.4f}",
        "",
        "[2] DEV OUT-OF-FOLD PROBABILITY CALIBRATION / RELIABILITY TABLE (HistGBDT)",
        f"Pooled OOF rows: {len(y_all):,} | Positives: {int(y_all.sum()):,} ({y_all.mean()*100:.3f}%) | Brier: {brier_score_loss(y_all, p_all):.6f}",
        cal_df.to_string(index=False),
    ]
    report_str = "\n".join(lines)
    out_txt.write_text(report_str + "\n", encoding="utf-8")
    print("\n" + report_str)
    print(f"\nSaved: {out_txt.relative_to(ROOT)}")


if __name__ == "__main__":
    main()