"""Supplementary Dev-only checks (strictly QDATE <= 2009-12-31; Test untouched)."""
from datetime import date
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SEED = 42

# Dynamically load exact functions/classes from src/08_preprocess.py & src/09_modelling.py
spec08 = importlib.util.spec_from_file_location("prep08", ROOT / "src" / "08_preprocess.py")
prep08 = importlib.util.module_from_spec(spec08)
spec08.loader.exec_module(prep08)

QuantileWinsorizer = prep08.QuantileWinsorizer

FOLDS = [
    ("Fold1_SL_1990_1993", date(1988, 12, 31), date(1990, 3, 31), date(1993, 12, 31)),
    ("Fold2_Calm_2000_2006", date(1998, 12, 31), date(2000, 3, 31), date(2006, 12, 31)),
    ("Fold3_GFC_2008_2009", date(1998 + 8, 12, 31), date(2008, 3, 31), date(2009, 12, 31)),
]

CONT_COLS = [
    "EQ_ASSET", "RBC1AAJ", "RBCRWAJ", "LOG_ASSET",
    "NPL_LOANS", "NONACCR_ASSET", "PD90_ASSET", "ORE_ASSET", "RES_LOANS",
    "EFFICIENCY",
    "ROA", "ROE", "NIMY", "ROA_Q_ANN", "NCO_RATE_Q", "PROV_RATE_Q",
    "LIQ_ASSET", "DEP_ASSET", "BRO_ASSET", "LOANS_ASSET", "RE_LOANS",
    "ASSET_G4", "DEP_G4", "EQ_ASSET_D4", "NPL_LOANS_D4",
    "ROA_D4", "LOANS_ASSET_D4", "ROA_MA4", "ROA_SD4",
]

DERIVED_MISS_SRC = ["ASSET_G4", "NPL_LOANS_D4", "ROA_MA4", "EFFICIENCY"]


def build_hgb_pipe(cont_cols: list[str], miss_cols: list[str]) -> Pipeline:
    # Reuse make_preprocess_pipeline from 08_preprocess if matching signature, else build identical
    num_pipe = Pipeline([
        ("winsor", QuantileWinsorizer(0.005, 0.995)),
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
    df_dev = (
        pl.scan_parquet(feat_path)
        .filter(pl.col("QDATE") <= date(2009, 12, 31))
        .collect()
    )
    assert df_dev["QDATE"].max() <= date(2009, 12, 31), "Leakage guard failed!"

    # Add the 4 derived *_MISS flags in memory if not already stored in features.parquet
    for c in DERIVED_MISS_SRC:
        mc = f"{c}_MISS"
        if mc not in df_dev.columns:
            df_dev = df_dev.with_columns(pl.col(c).is_null().cast(pl.Int8).alias(mc))

    miss_cols = [c for c in df_dev.columns if c.endswith("_MISS") and c != "P3ASSET_MISS"]
    print(f"Loaded Dev-only panel: {df_dev.height:,} rows | {len(CONT_COLS)} cont + {len(miss_cols)} miss flags = {len(CONT_COLS)+len(miss_cols)} inputs")

    pdf_base = df_dev.select(["QDATE", "y"] + CONT_COLS + miss_cols).to_pandas()

    # In-memory variant: pre-1990 zeros in RBC1AAJ / RBCRWAJ -> NaN + *_MISS flags
    pdf_alt = pdf_base.copy()
    pre1990 = pd.to_datetime(pdf_alt["QDATE"]).dt.year < 1990
    for col in ["RBC1AAJ", "RBCRWAJ"]:
        mask_zero = pre1990 & (pdf_alt[col] == 0.0)
        n_replaced = int(mask_zero.sum())
        pdf_alt.loc[mask_zero, col] = np.nan
        pdf_alt[f"{col}_MISS"] = pdf_alt[col].isna().astype(np.int8)
        print(f"In-memory pre-1990 {col} == 0 -> NaN: {n_replaced:,} rows")

    alt_miss_cols = miss_cols + [c for c in ["RBC1AAJ_MISS", "RBCRWAJ_MISS"] if c not in miss_cols]

    rows_comp = []
    oof_y, oof_p_base = [], []

    for fname, tr_end, va_start, va_end in FOLDS:
        assert (va_start - tr_end).days > 400, f"Purge gap < 4 quarters on {fname}"
        qdates = pd.to_datetime(pdf_base["QDATE"]).dt.date
        tr_mask = qdates <= tr_end
        va_mask = (qdates >= va_start) & (qdates <= va_end)

        y_tr = pdf_base.loc[tr_mask, "y"].to_numpy()
        y_va = pdf_base.loc[va_mask, "y"].to_numpy()

        pipe_base = build_hgb_pipe(CONT_COLS, miss_cols)
        pipe_base.fit(pdf_base.loc[tr_mask, CONT_COLS + miss_cols], y_tr)
        p_base = pipe_base.predict_proba(pdf_base.loc[va_mask, CONT_COLS + miss_cols])[:, 1]

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

    df_comp = pd.DataFrame(rows_comp)
    mean_base = df_comp["pr_auc_baseline"].mean()
    mean_alt = df_comp["pr_auc_rbc_null"].mean()
    y_all = np.concatenate(oof_y)
    p_all = np.concatenate(oof_p_base)
    pooled_pr_base = average_precision_score(y_all, p_all)

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
    cal_df["pred_to_actual_ratio"] = (
        cal_df["mean_pred_prob"] / cal_df["actual_fail_rate"].replace(0, np.nan)
    ).round(2)

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
        f"Pooled OOF PR-AUC (baseline sanity check vs Table 11): {pooled_pr_base:.4f}",
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