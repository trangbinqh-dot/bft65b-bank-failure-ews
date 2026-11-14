"""Part 6 helper: rebuild every report figure from the CSV files saved in outputs/.

Reads only files that earlier scripts already wrote (no raw data, no model refit, no test
re-evaluation). Run from the repository root:

    python src/10_make_figures.py

Figures are written to outputs/figures/*.png. A figure whose input CSV is missing is skipped.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path("outputs")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# colour-blind-safe palette (Okabe-Ito subset)
C_BLUE, C_ORANGE, C_GREEN, C_GREY, C_RED = "#0072B2", "#E69F00", "#009E73", "#7F7F7F", "#D55E00"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "axes.axisbelow": True,
})


def read(name):
    p = OUT / name
    if not p.exists():
        print(f"[skip] {p} not found")
        return None
    return pd.read_csv(p)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok]   outputs/figures/{name}")


# ---------------------------------------------------------------- Part 1: storage
df = read("part1_storage_benchmark.csv")
if df is not None:
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.0))
    ax[0].barh(df["Layout"], df["Size (MB)"], color=C_BLUE)
    ax[0].set_xlabel("Size on disk (MB)")
    ax[0].set_title("Storage footprint")
    ax[0].invert_yaxis()
    y = range(len(df))
    h = 0.38
    ax[1].barh([i - h / 2 for i in y], df["Full read (s)"], h, label="Full read", color=C_BLUE)
    ax[1].barh([i + h / 2 for i in y], df["5 cols + year>=2008 (s)"], h,
               label="5 columns + year >= 2008", color=C_ORANGE)
    ax[1].set_yticks(list(y))
    ax[1].set_yticklabels([])  # share row order with left panel to avoid label collision
    ax[1].invert_yaxis()
    ax[1].set_xlabel("Read time (s)")
    ax[1].set_title("Read performance")
    ax[1].legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92, edgecolor="none")
    save(fig, "fig1_storage_benchmark.png")

# ---------------------------------------------------------------- Part 2: banks per quarter
df = read("part2_banks_per_quarter.csv")
if df is not None:
    d = pd.to_datetime(df["REPDTE"].astype(str), format="%Y%m%d", errors="coerce")
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.plot(d, df["n_banks"], color=C_BLUE, lw=1.6)
    ax.set_ylabel("Reporting banks")
    ax.set_title("Banks in the Call Report panel per quarter")
    save(fig, "fig2_banks_per_quarter.png")

# ---------------------------------------------------------------- Part 3: engines
df = read("part3_polars_vs_pandas.csv")
if df is not None:
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.8))
    ax[0].bar(df["method"], df["median_seconds"], color=C_BLUE)
    ax[0].set_ylabel("Median wall time (s)")
    ax[0].set_title("Speed (median of repeated runs)")
    ax[1].bar(df["method"], df["peak_RAM_MB"], color=C_ORANGE)
    ax[1].set_ylabel("Peak RAM (MB)")
    ax[1].set_title("Peak memory")
    for a in ax:
        a.tick_params(axis="x", rotation=15)
    save(fig, "fig3_polars_vs_pandas.png")

df = read("part3_scaling.csv")
if df is not None:
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.8))
    colors = {"polars_lazy": C_BLUE, "pandas": C_ORANGE}
    for m, g in df.groupby("method"):
        g = g.sort_values("panel_rows")
        ax[0].plot(g["panel_rows"] / 1e6, g["median_seconds"], "o-", label=m, color=colors.get(m, C_GREY))
        ax[1].plot(g["panel_rows"] / 1e6, g["peak_RAM_MB"], "o-", label=m, color=colors.get(m, C_GREY))
    ax[0].set_xlabel("Panel rows (millions)")
    ax[0].set_ylabel("Median time (s)")
    ax[0].set_title("Scaling: time")
    ax[1].set_xlabel("Panel rows (millions)")
    ax[1].set_ylabel("Peak RAM (MB)")
    ax[1].set_title("Scaling: memory")
    ax[0].legend(loc="upper left", frameon=False)
    save(fig, "fig4_scaling.png")

# ---------------------------------------------------------------- Part 5: CV
df = read("part5_cv_results.csv")
if df is not None:
    pv = df.pivot(index="fold", columns="model", values="PR_AUC")
    pv = pv.reindex(sorted(pv.index))
    fig, ax = plt.subplots(figsize=(8.5, 3.9))
    pv.plot.bar(ax=ax, color=[C_GREY, C_ORANGE, C_BLUE][: pv.shape[1]], width=0.72)
    ax.set_ylabel("PR-AUC")
    ax.set_xlabel("")
    ax.set_ylim(0, pv.to_numpy().max() * 1.18)
    ax.set_title("Temporal cross-validation on the development period (PR-AUC)")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(loc="upper left", frameon=False, title=None)
    save(fig, "fig5_cv_pr_auc.png")

# ---------------------------------------------------------------- Part 5: threshold
df = read("part5_threshold_curve.csv")
if df is not None:
    i = df["total_cost_USD_M"].idxmin()
    tau = df.loc[i, "threshold"]
    min_cost = df.loc[i, "total_cost_USD_M"]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))
    ax[0].plot(df["threshold"], df["total_cost_USD_M"], color=C_BLUE, lw=1.8)
    ax[0].axvline(tau, color=C_RED, ls="--", lw=1.2)
    ax[0].annotate(f"tau* = {tau:.3f}\n(${min_cost:.1f}M)", (tau, min_cost),
                   xytext=(tau + 0.12, min_cost + (df["total_cost_USD_M"].max() - min_cost) * 0.25),
                   textcoords="data", color=C_RED, fontweight="bold",
                   arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.0))
    ax[0].set_xlabel("Alarm threshold")
    ax[0].set_ylabel("Total cost on dev out-of-fold (USD M)")
    ax[0].set_title("Cost-based threshold (development data only)")
    ax[1].plot(df["threshold"], df["recall"], label="Recall", color=C_GREEN, lw=1.8)
    ax[1].plot(df["threshold"], df["precision"], label="Precision", color=C_ORANGE, lw=1.8)
    ax[1].axvline(tau, color=C_RED, ls="--", lw=1.2)
    ax[1].set_xlabel("Alarm threshold")
    ax[1].set_ylabel("Rate")
    ax[1].set_title("Recall / precision trade-off")
    ax[1].legend(loc="center right", frameon=False)
    save(fig, "fig6_threshold_cost.png")

# ---------------------------------------------------------------- Part 5: importance
df = read("part5_feature_importance.csv")
if df is not None:
    top = df.sort_values("pr_auc_drop_mean", ascending=False).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.barh(top["feature"], top["pr_auc_drop_mean"], xerr=top["pr_auc_drop_std"], color=C_BLUE,
            error_kw=dict(ecolor=C_GREY, lw=1))
    ax.set_xlabel("Drop in PR-AUC when feature is permuted (dev fold 3)")
    ax.set_title("Top 15 features (Permutation Importance)")
    save(fig, "fig7_feature_importance.png")

# ---------------------------------------------------------------- Part 5: test
df = read("part5_test_evaluation.csv")
if df is not None:
    mask = df["threshold_rule"].astype(str).str.lower().str.contains("lock") | (df["tau"] < 0.25)
    d = df[mask].drop_duplicates(subset=["model"]).set_index("model")
    rec_pct = d["Recall_row"] * 100 if d["Recall_row"].max() <= 1.0 else d["Recall_row"]
    no_alarm = d["NoAlarm_Cost_USD_M"].iloc[0]
    fig, ax = plt.subplots(1, 3, figsize=(11.2, 3.9))
    ax[0].bar(d.index, d["PR_AUC"], color=C_BLUE, width=0.55)
    ax[0].set_ylim(0, max(0.7, d["PR_AUC"].max() * 1.18))
    ax[0].set_title("Test PR-AUC")
    ax[1].bar(d.index, rec_pct, color=C_GREEN, width=0.55)
    ax[1].set_ylim(0, 105)
    ax[1].set_title("Test recall at locked tau* (%)")
    ax[2].bar(d.index, d["Test_Cost_USD_M"], color=C_ORANGE, width=0.55)
    ax[2].axhline(no_alarm, color=C_RED, ls="--", lw=1.2)
    ax[2].set_ylim(0, no_alarm * 1.18)
    ax[2].annotate(f"no-alarm: ${no_alarm:.1f}M", (0, no_alarm),
                   xytext=(4, 5), textcoords="offset points", color=C_RED, fontsize=8.5, fontweight="bold")
    ax[2].set_title("Test cost (USD M, assumed unit costs)")
    for a in ax:
        a.tick_params(axis="x", rotation=18)
    save(fig, "fig8_test_evaluation.png")

print("done")
