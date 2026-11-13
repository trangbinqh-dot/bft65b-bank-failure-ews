import gc
import importlib
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import tee_log

prep_mod = importlib.import_module("08_preprocess")
NUM_FEATURES = prep_mod.NUM_FEATURES
ALL_COLS = prep_mod.ALL_COLS
DEV_END = prep_mod.DEV_END
TEST_START = prep_mod.TEST_START
QuantileWinsorizer = prep_mod.QuantileWinsorizer
add_derived_miss_flags = prep_mod.add_derived_miss_flags
make_preprocess_pipeline = prep_mod.make_preprocess_pipeline

SEED = 42
BASELINE_COLS = ["EQ_ASSET", "NPL_LOANS", "ROA", "LIQ_ASSET"]
BASELINE_IDX = [ALL_COLS.index(c) for c in BASELINE_COLS]

# 3 Expanding-Window Temporal CV Folds strictly inside DEV (<= 2009-12-31)
# Every fold enforces a 4-quarter purge gap between train_end and val_start!
CV_FOLDS = [
    ("Fold1_SL_1990_1993", date(1988, 12, 31), date(1990, 3, 31), date(1993, 12, 31)),
    ("Fold2_Calm_2000_2006", date(1998, 12, 31), date(2000, 3, 31), date(2006, 12, 31)),
    ("Fold3_GFC_2008_2009", date(2006, 12, 31), date(2008, 3, 31), date(2009, 12, 31)),
]


def recall_at_top_k(y_true: np.ndarray, y_prob: np.ndarray, frac: float = 0.05) -> float:
    n_top = max(1, int(round(len(y_prob) * frac)))
    top_idx = np.argpartition(y_prob, -n_top)[-n_top:]
    total_pos = y_true.sum()
    return float(y_true[top_idx].sum() / total_pos) if total_pos > 0 else 0.0


def build_models() -> dict[str, Pipeline]:
    base_prep = ColumnTransformer(
        transformers=[
            (
                "num4",
                Pipeline([
                    ("winsorize", QuantileWinsorizer(0.005, 0.995)),
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]),
                BASELINE_IDX,
            )
        ],
        remainder="drop",
    )
    return {
        "baseline_camels4": Pipeline([
            ("prep", base_prep),
            ("clf", LogisticRegression(C=1.0, max_iter=500, random_state=SEED)),
        ]),
        "logistic_l2_full": Pipeline([
            ("prep", make_preprocess_pipeline()),
            ("clf", LogisticRegression(C=0.1, max_iter=500, random_state=SEED)),
        ]),
        "hist_gbdt_full": Pipeline([
            ("prep", make_preprocess_pipeline()),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=150,
                    learning_rate=0.05,
                    max_leaf_nodes=31,
                    min_samples_leaf=100,
                    l2_regularization=1.0,
                    random_state=SEED,
                ),
            ),
        ]),
    }


def estimate_supervisory_costs_on_dev() -> tuple[float, float, dict]:
    """Estimate c_FN and c_FP (in $ thousands) strictly from DEV period (<= 2009-12-31)."""
    fc = (
        pl.read_parquet("data_processed/failure_costs.parquet")
        .with_columns(pl.col("COST").cast(pl.Float64, strict=False))
        .filter((pl.col("FD") <= DEV_END) & pl.col("COST").is_not_null() & (pl.col("COST") > 0))
    )
    med_dif_loss_k = float(fc["COST"].median())
    mean_dif_loss_k = float(fc["COST"].mean())

    # Avoidable DIF loss per missed positive bank-quarter:
    # 20% loss mitigation via early Prompt Corrective Action shared across 4 warning quarters
    c_fn_k = 0.20 * med_dif_loss_k / 4.0

    # False-alarm special examination + compliance cost anchored to DEV bank quarterly operating expense
    opex = (
        pl.scan_parquet("data_processed/panel_clean.parquet")
        .filter((pl.col("QDATE") <= DEV_END) & pl.col("NONIX_Q").is_not_null() & (pl.col("NONIX_Q") > 0))
        .select(pl.col("NONIX_Q").cast(pl.Float64).median())
        .collect()
        .item()
    )
    c_fp_k = 0.15 * float(opex)

    info = {
        "n_dev_failed_with_cost": fc.height,
        "median_dif_cost_k": round(med_dif_loss_k, 1),
        "mean_dif_cost_k": round(mean_dif_loss_k, 1),
        "median_bank_nonix_q_k": round(float(opex), 1),
        "c_fn_k": round(c_fn_k, 2),
        "c_fp_k": round(c_fp_k, 2),
        "cost_ratio_fn_over_fp": round(c_fn_k / c_fp_k, 2),
        "bayes_theoretical_tau": round(c_fp_k / (c_fp_k + c_fn_k), 4),
    }
    return c_fn_k, c_fp_k, info


if __name__ == "__main__":
    Path("outputs").mkdir(exist_ok=True)
    tee_log("outputs/part5_modelling_report.txt")
    np.random.seed(SEED)

    print("=== PART 5: MODELLING, TEMPORAL VALIDATION & COST-BASED THRESHOLDING ===")
    df = add_derived_miss_flags(pl.read_parquet("data_processed/features.parquet")).sort("QDATE", "CERT")

    dev_df = df.filter(pl.col("QDATE") <= DEV_END)
    test_df = df.filter(pl.col("QDATE") >= TEST_START)
    del df
    gc.collect()

    X_dev = dev_df.select(ALL_COLS).to_numpy()
    y_dev = dev_df["y"].to_numpy()
    q_dev = dev_df["QDATE"].to_list()

    X_test = test_df.select(ALL_COLS).to_numpy()
    y_test = test_df["y"].to_numpy()
    cert_test = test_df["CERT"].to_numpy()
    print(f"Loaded DEV: {X_dev.shape}, y=1: {y_dev.sum():,} ({y_dev.mean()*100:.3f}%)")
    print(f"Loaded TEST (locked until Step 4): {X_test.shape}, y=1: {y_test.sum():,} ({y_test.mean()*100:.3f}%)")

    # ------------------------------------------------------------------
    # 1. Expanding-Window Temporal Cross-Validation on DEV (4-qtr purge)
    # ------------------------------------------------------------------
    q_arr = np.array(q_dev)
    cv_rows = []
    oof_store = {m: {"y": [], "p": []} for m in build_models()}

    print("\n[1] EXPANDING-WINDOW TEMPORAL CV ON DEV (4-QUARTER PURGE GAP PER FOLD)")
    for fold_name, tr_end, va_start, va_end in CV_FOLDS:
        # Verify >= 4 quarters gap (~365+ days between tr_end and va_start)
        assert (va_start - tr_end).days > 400, "Purge gap must exceed 4 quarters!"
        tr_mask = q_arr <= tr_end
        va_mask = (q_arr >= va_start) & (q_arr <= va_end)

        X_tr, y_tr = X_dev[tr_mask], y_dev[tr_mask]
        X_va, y_va = X_dev[va_mask], y_dev[va_mask]
        print(f"\n  {fold_name}: Train<={tr_end} ({len(y_tr):,} rows, {y_tr.sum():,} pos) "
              f"| Val {va_start}..{va_end} ({len(y_va):,} rows, {y_va.sum():,} pos)")

        models = build_models()
        for mname, pipe in models.items():
            t0 = time.perf_counter()
            pipe.fit(X_tr, y_tr)
            prob = pipe.predict_proba(X_va)[:, 1]
            dt = time.perf_counter() - t0

            pr_auc = average_precision_score(y_va, prob)
            roc_auc = roc_auc_score(y_va, prob)
            rec_top5 = recall_at_top_k(y_va, prob, 0.05)
            brier = brier_score_loss(y_va, prob)

            oof_store[mname]["y"].append(y_va)
            oof_store[mname]["p"].append(prob)
            cv_rows.append({
                "fold": fold_name,
                "model": mname,
                "train_rows": int(len(y_tr)),
                "val_rows": int(len(y_va)),
                "val_pos": int(y_va.sum()),
                "PR_AUC": round(pr_auc, 4),
                "ROC_AUC": round(roc_auc, 4),
                "Recall_Top5pct": round(rec_top5, 4),
                "Brier": round(brier, 5),
                "fit_sec": round(dt, 2),
            })
            print(f"    {mname:<18} -> PR-AUC={pr_auc:.4f} | ROC-AUC={roc_auc:.4f} | Rec@Top5%={rec_top5:.4f} ({dt:.1f}s)")
        gc.collect()

    # Pooled OOF & Mean across folds
    print("\n  --- POOLED OUT-OF-FOLD & MEAN TEMPORAL CV SUMMARY ---")
    summary_cv = []
    for mname in oof_store:
        yy = np.concatenate(oof_store[mname]["y"])
        pp = np.concatenate(oof_store[mname]["p"])
        sub = [r for r in cv_rows if r["model"] == mname]
        mean_pr = float(np.mean([r["PR_AUC"] for r in sub]))
        mean_roc = float(np.mean([r["ROC_AUC"] for r in sub]))
        oof_pr = average_precision_score(yy, pp)
        oof_roc = roc_auc_score(yy, pp)
        oof_top5 = recall_at_top_k(yy, pp, 0.05)
        summary_cv.append({
            "model": mname,
            "mean_fold_PR_AUC": round(mean_pr, 4),
            "mean_fold_ROC_AUC": round(mean_roc, 4),
            "pooled_OOF_PR_AUC": round(oof_pr, 4),
            "pooled_OOF_ROC_AUC": round(oof_roc, 4),
            "pooled_OOF_Rec_Top5pct": round(oof_top5, 4),
        })
    df_cv = pd.DataFrame(cv_rows)
    df_cv_sum = pd.DataFrame(summary_cv).sort_values("mean_fold_PR_AUC", ascending=False)
    print(df_cv_sum.to_string(index=False))
    df_cv.to_csv("outputs/part5_cv_results.csv", index=False)

    champion_name = df_cv_sum.iloc[0]["model"]
    print(f"\nSelected Champion Model on DEV CV: {champion_name}")

    # ------------------------------------------------------------------
    # 2. Data-Driven Cost Estimation & Alarm Threshold Selection on DEV
    # ------------------------------------------------------------------
    c_fn, c_fp, cost_info = estimate_supervisory_costs_on_dev()
    print("\n[2] DATA-DRIVEN SUPERVISORY COST ESTIMATION (DEV <= 2009-12-31 ONLY)")
    for k, v in cost_info.items():
        print(f"  {k:<26}: {v}")

    y_oof = np.concatenate(oof_store[champion_name]["y"])
    p_oof = np.concatenate(oof_store[champion_name]["p"])

    grid = np.unique(np.round(np.concatenate([
        np.linspace(0.005, 0.10, 96),
        np.linspace(0.105, 0.50, 80),
    ]), 4))

    curve_rows = []
    for tau in grid:
        pred = (p_oof >= tau).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_oof, pred).ravel()
        tot_cost_m = (fn * c_fn + fp * c_fp) / 1000.0  # in $ Millions
        curve_rows.append({
            "threshold": float(tau),
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
            "precision": round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0,
            "recall": round(tp / (tp + fn), 4),
            "f1": round(f1_score(y_oof, pred, zero_division=0), 4),
            "flag_rate_pct": round((tp + fp) / len(y_oof) * 100, 3),
            "total_cost_USD_M": round(tot_cost_m, 2),
        })
    df_curve = pd.DataFrame(curve_rows)
    df_curve.to_csv("outputs/part5_threshold_curve.csv", index=False)

    best_row = df_curve.loc[df_curve["total_cost_USD_M"].idxmin()]
    tau_star = float(best_row["threshold"])
    no_alarm_oof_m = (y_oof.sum() * c_fn) / 1000.0
    print(f"\nOptimal Threshold on DEV OOF: tau* = {tau_star:.4f}")
    print(f"  DEV OOF at tau*: Cost=${best_row['total_cost_USD_M']:.2f}M (vs No-Alarm ${no_alarm_oof_m:.2f}M) "
          f"| Recall={best_row['recall']:.4f} | Precision={best_row['precision']:.4f} | F1={best_row['f1']:.4f} "
          f"| Flagged={best_row['flag_rate_pct']:.2f}%")

    print("\n  --- COST-RATIO SENSITIVITY ON DEV OOF (c_FN / c_FP) ---")
    for ratio in (5, 10, round(c_fn / c_fp, 1), 20, 50):
        costs_r = [((r["FN"] * ratio * c_fp + r["FP"] * c_fp) / 1000.0, r) for r in curve_rows]
        _, br = min(costs_r, key=lambda x: x[0])
        print(f"  Ratio {ratio:>4}:1 -> optimal tau={br['threshold']:.4f} | Recall={br['recall']:.3f} "
              f"| Precision={br['precision']:.3f} | Flagged={br['flag_rate_pct']:.2f}%")

    # ------------------------------------------------------------------
    # 3. Feature Importance on Fold-3 Validation (DEV-only, pre-Test)
    # ------------------------------------------------------------------
    print("\n[3] PERMUTATION FEATURE IMPORTANCE ON DEV FOLD-3 (PR-AUC DROP)")
    f3_tr = q_arr <= date(2006, 12, 31)
    f3_va = (q_arr >= date(2008, 3, 31)) & (q_arr <= date(2009, 12, 31))
    champ_f3 = build_models()[champion_name]
    champ_f3.fit(X_dev[f3_tr], y_dev[f3_tr])
    perm = permutation_importance(
        champ_f3, X_dev[f3_va], y_dev[f3_va],
        scoring="average_precision", n_repeats=3, random_state=SEED,
    )
    df_imp = pd.DataFrame({
        "feature": ALL_COLS,
        "pr_auc_drop_mean": np.round(perm.importances_mean, 5),
        "pr_auc_drop_std": np.round(perm.importances_std, 5),
    }).sort_values("pr_auc_drop_mean", ascending=False)
    df_imp.to_csv("outputs/part5_feature_importance.csv", index=False)
    print(df_imp.head(12).to_string(index=False))
    del champ_f3
    gc.collect()

    # ------------------------------------------------------------------
    # 4. SINGLE FINAL EVALUATION ON LOCKED TEST PERIOD (2011Q1 - 2025Q2)
    # ------------------------------------------------------------------
    print(f"\n[4] SINGLE EVALUATION ON TEST SET (>= {TEST_START}, LOCKED tau* = {tau_star:.4f})")
    final_models = build_models()
    test_rows = []
    no_alarm_test_m = (y_test.sum() * c_fn) / 1000.0

    failed_test_certs = set(cert_test[y_test == 1])
    n_failed_banks_test = len(failed_test_certs)

    for mname, pipe in final_models.items():
        t0 = time.perf_counter()
        pipe.fit(X_dev, y_dev)
        p_test = pipe.predict_proba(X_test)[:, 1]
        fit_s = time.perf_counter() - t0

        pr_auc = average_precision_score(y_test, p_test)
        roc_auc = roc_auc_score(y_test, p_test)
        rec_top5 = recall_at_top_k(y_test, p_test, 0.05)
        brier = brier_score_loss(y_test, p_test)

        for tag, th in ((f"locked_tau_{tau_star:.4f}", tau_star), ("naive_tau_0.50", 0.50)):
            pred = (p_test >= th).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
            cost_m = (fn * c_fn + fp * c_fp) / 1000.0
            caught_banks = len(set(cert_test[(y_test == 1) & (pred == 1)]))
            test_rows.append({
                "model": mname,
                "threshold_rule": tag,
                "tau": th,
                "PR_AUC": round(pr_auc, 4),
                "ROC_AUC": round(roc_auc, 4),
                "Recall_Top5pct": round(rec_top5, 4),
                "Brier": round(brier, 5),
                "TP": int(tp),
                "FP": int(fp),
                "FN": int(fn),
                "TN": int(tn),
                "Precision": round(precision_score(y_test, pred, zero_division=0), 4),
                "Recall_row": round(recall_score(y_test, pred, zero_division=0), 4),
                "F1": round(f1_score(y_test, pred, zero_division=0), 4),
                "Flag_rate_pct": round((tp + fp) / len(y_test) * 100, 3),
                "Banks_caught_of_199": f"{caught_banks}/{n_failed_banks_test} ({caught_banks/n_failed_banks_test*100:.1f}%)",
                "Test_Cost_USD_M": round(cost_m, 2),
                "NoAlarm_Cost_USD_M": round(no_alarm_test_m, 2),
                "fit_sec": round(fit_s, 2),
            })

    df_test = pd.DataFrame(test_rows)
    df_test.to_csv("outputs/part5_test_evaluation.csv", index=False)
    print(df_test.to_string(index=False))
    print("\nSaved all Part 5 tables to outputs/ (Test set evaluated exactly ONCE).")