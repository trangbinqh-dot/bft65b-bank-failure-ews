"""Part 6 helper: rebuild ALL report figures directly from saved files in outputs/.

Reads only files that earlier scripts already wrote (no raw data, no model refit, no test
re-evaluation). Run from the repository root:

    python src/10_make_figures.py

Figures are written to outputs/figures/fig1_*.png ... fig9_*.png.
"""
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(os.environ.get("BFT65B_OUT", "outputs"))
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Colour-blind-safe Okabe-Ito palette
C_NAVY = "#1B365D"
C_BLUE = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN = "#009E73"
C_GREY = "#7F8C8D"
C_RED = "#D55E00"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "axes.axisbelow": True,
})


def read_csv(name):
    p = OUT / name
    if not p.exists():
        print(f"[skip] {p} not found")
        return None
    return pd.read_csv(p)


def parse_polars_table(path, col_key, col_val):
    """Parse ASCII/Unicode Polars table from saved .txt report into dict."""
    p = OUT / path
    if not p.exists():
        print(f"[skip] {p} not found")
        return None
    txt = p.read_text(encoding="utf-8", errors="replace")
    rows = {}
    in_target = False
    for line in txt.splitlines():
        if col_key in line and col_val in line:
            in_target = True
            continue
        if in_target:
            if line.startswith("└") or (line.strip() == "" and len(rows) > 0):
                break
            parts = [c.strip() for c in re.split(r"[│┆|]", line) if c.strip()]
            if len(parts) >= 2 and parts[0] not in ("---", "str", col_key):
                try:
                    rows[parts[0]] = float(parts[-1])
                except ValueError:
                    pass
    return rows if rows else None


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok]   outputs/figures/{name}")


# ---------------------------------------------------------------- Fig 0: Architecture diagram
fig, ax = plt.subplots(figsize=(10.6, 2.2))
ax.axis("off")
steps = [
    ("1. Acquisition", "FDIC API (1984Q1-2026Q2)\nResumable + Manifest"),
    ("2. Storage", "Parquet + Zstd\nSchema Harmonisation"),
    ("3. Quality Audit", "Schema, Nulls, Anomalies\nYTD De-cumulation (*_Q)"),
    ("4. Clean Panel", "Point-in-Time y(q, q+4]\nIsolate COST & >=FAILDATE"),
    ("5. Scalable Engine", "Polars Lazy vs. Pandas\nPySpark Catalyst Plan"),
    ("6. EWS Modelling", "29 CAMELS + 10 Flags\nDev CV -> Single Test Run"),
]
w, h, gap = 0.132, 0.66, 0.036
for idx, (title, body) in enumerate(steps):
    x = 0.01 + idx * (w + gap)
    hdr = mpatches.FancyBboxPatch((x, 0.52), w, 0.30, boxstyle="round,pad=0.008", fc=C_NAVY, ec=C_NAVY)
    box = mpatches.FancyBboxPatch((x, 0.12), w, 0.42, boxstyle="round,pad=0.008", fc="#F8FAFC", ec=C_NAVY, lw=1.1)
    ax.add_patch(box)
    ax.add_patch(hdr)
    ax.text(x + w / 2, 0.67, title, ha="center", va="center", color="white", fontweight="bold", fontsize=8.8)
    ax.text(x + w / 2, 0.32, body, ha="center", va="center", color="#1E293B", fontsize=7.6)
    if idx < len(steps) - 1:
        ax.annotate("", xy=(x + w + gap - 0.002, 0.47), xytext=(x + w + 0.002, 0.47),
                    arrowprops=dict(arrowstyle="->", color=C_NAVY, lw=1.5))
ax.set_xlim(0, 1)
ax.set_ylim(0, 0.95)
ax.set_title("End-to-End FDIC Bank Failure Early-Warning Pipeline Architecture", fontweight="bold", fontsize=10.5, pad=4)
save(fig, "fig0_pipeline_architecture.png")

# ---------------------------------------------------------------- Fig 1: Part 1 storage
df = read_csv("part1_storage_benchmark.csv")
if df is not None:
    fig, ax = plt.subplots(1, 2, figsize=(10.8, 3.3))
    ax[0].barh(df["Layout"], df["Size (MB)"], color=C_BLUE, height=0.62)
    ax[0].set_xlabel("Size on disk (MB)")
    ax[0].set_title("(a) Storage footprint")
    ax[0].invert_yaxis()
    y = range(len(df))
    bh = 0.36
    ax[1].barh([i - bh / 2 for i in y], df["Full read (s)"], bh, label="Full read", color=C_BLUE)
    ax[1].barh([i + bh / 2 for i in y], df["5 cols + year>=2008 (s)"], bh,
               label="5 columns + year >= 2008", color=C_ORANGE)
    ax[1].set_yticks(list(y))
    ax[1].set_yticklabels([])
    ax[1].invert_yaxis()
    ax[1].set_xlabel("Read time (s)")
    ax[1].set_title("(b) Read performance")
    ax[1].legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92, edgecolor="none")
    save(fig, "fig1_storage_benchmark.png")

# ---------------------------------------------------------------- Fig 2: Part 2 banks per quarter + split
df = read_csv("part2_banks_per_quarter.csv")
if df is not None:
    d = pd.to_datetime(df["REPDTE"].astype(str), format="%Y%m%d", errors="coerce")
    fig, ax = plt.subplots(figsize=(9.6, 3.1))
    ax.axvspan(pd.Timestamp("1984-03-31"), pd.Timestamp("2009-12-31"), color="#EBF5FB", alpha=0.7, label="Dev (1984Q1-2009Q4)")
    ax.axvspan(pd.Timestamp("2010-01-01"), pd.Timestamp("2010-12-31"), color="#FEF9E7", alpha=0.95, label="4-Qtr Purge Gap (2010)")
    ax.axvspan(pd.Timestamp("2011-01-01"), pd.Timestamp("2025-06-30"), color="#E8F8F5", alpha=0.7, label="Test (2011Q1-2025Q2)")
    ax.axvspan(pd.Timestamp("2025-07-01"), pd.Timestamp("2026-06-30"), color="#EBEDEF", alpha=0.8, label="Right-censored (>2025Q2)")
    ax.plot(d, df["n_banks"], color=C_NAVY, lw=1.8, label="Reporting banks")
    idx_max = df["n_banks"].idxmax()
    ax.annotate(f"Peak: {int(df.loc[idx_max, 'n_banks']):,}", (d.iloc[idx_max], df.loc[idx_max, "n_banks"]),
                xytext=(25, -4), textcoords="offset points", fontsize=8.5, fontweight="bold", color=C_NAVY,
                arrowprops=dict(arrowstyle="->", color=C_NAVY, lw=0.9))
    ax.annotate(f"End: {int(df['n_banks'].iloc[-1]):,}", (d.iloc[-1], df["n_banks"].iloc[-1]),
                xytext=(-75, 16), textcoords="offset points", fontsize=8.5, fontweight="bold", color=C_NAVY,
                arrowprops=dict(arrowstyle="->", color=C_NAVY, lw=0.9))
    ax.set_ylabel("Reporting banks (CERT)")
    ax.set_xlabel("Report quarter (REPDTE)")
    ax.set_title("Unbalanced Call Report Panel Size (1984Q1-2026Q2) and Temporal Split (Option B)")
    ax.legend(loc="upper right", fontsize=8, frameon=True, facecolor="white", framealpha=0.9)
    save(fig, "fig2_banks_per_quarter.png")

# ---------------------------------------------------------------- Fig 3: Part 3 Polars vs Pandas
df = read_csv("part3_polars_vs_pandas.csv")
if df is not None:
    fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.1))
    ax[0].bar(df["method"], df["median_seconds"], color=C_BLUE, width=0.55)
    ax[0].set_ylabel("Median wall time (s)")
    ax[0].set_title("(a) Execution speed (Dev panel)")
    ax[1].bar(df["method"], df["peak_RAM_MB"], color=C_ORANGE, width=0.55)
    ax[1].set_ylabel("Peak Working Set RAM (MB)")
    ax[1].set_title("(b) Peak memory")
    for a in ax:
        a.tick_params(axis="x", rotation=12)
    save(fig, "fig3_polars_vs_pandas.png")

# ---------------------------------------------------------------- Fig 4: Part 3 Scaling
df = read_csv("part3_scaling.csv")
if df is not None:
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.1))
    colors = {"polars_lazy": C_BLUE, "pandas": C_ORANGE}
    for m, g in df.groupby("method"):
        g = g.sort_values("panel_rows")
        ax[0].plot(g["panel_rows"] / 1e6, g["median_seconds"], "o-", lw=1.8, label=m, color=colors.get(m, C_GREY))
        ax[1].plot(g["panel_rows"] / 1e6, g["peak_RAM_MB"], "o-", lw=1.8, label=m, color=colors.get(m, C_GREY))
    ax[0].set_xlabel("Panel rows (millions)")
    ax[0].set_ylabel("Median time (s)")
    ax[0].set_title("(a) Scaling: execution time (1x-3x)")
    ax[1].set_xlabel("Panel rows (millions)")
    ax[1].set_ylabel("Peak RAM (MB)")
    ax[1].set_title("(b) Scaling: peak memory (1x-3x)")
    ax[0].legend(loc="upper left", frameon=False)
    save(fig, "fig4_scaling.png")

# ---------------------------------------------------------------- Fig 5: Part 4 Winsorization correlations
pre_corr = parse_polars_table("part4_features_report.txt", "column", "corr_with_y_dev")
post_corr = parse_polars_table("part4_preprocess_report.txt", "feature", "dev_post_win_corr_y")
if pre_corr and post_corr:
    cdf = pd.DataFrame({"pre": pd.Series(pre_corr), "post": pd.Series(post_corr)}).dropna()
    cdf["abs_post"] = cdf["post"].abs()
    cdf = cdf.sort_values("abs_post", ascending=False).head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    y = np.arange(len(cdf))
    bh = 0.38
    ax.barh(y - bh / 2, cdf["pre"], bh, color=C_GREY, alpha=0.75, label="Raw feature (pre-winsorization)")
    ax.barh(y + bh / 2, cdf["post"], bh, color=C_NAVY, label="Post-pipeline on Dev (0.5%-99.5% winsorized + imputed)")
    ax.axvline(0, color="#334155", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(cdf.index)
    ax.set_xlim(-0.34, 0.33)
    ax.set_xlabel("Pearson correlation (r) with failure label y on Dev period (n = 1,295,349)")
    ax.set_title("Signal Recovery After Training-Only Winsorization (Top 12 CAMELS Features)", pad=26)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2, fontsize=8.2, frameon=True, facecolor="white", edgecolor="#CBD5E1")
    save(fig, "fig5_winsorization_corr.png")

# ---------------------------------------------------------------- Fig 6: Part 5 CV PR-AUC
df = read_csv("part5_cv_results.csv")
if df is not None:
    pv = df.pivot(index="fold", columns="model", values="PR_AUC")
    pv = pv.reindex(sorted(pv.index))
    fig, ax = plt.subplots(figsize=(8.6, 3.1))
    pv.plot.bar(ax=ax, color=[C_GREY, C_ORANGE, C_BLUE][: pv.shape[1]], width=0.70)
    ax.set_ylabel("PR-AUC")
    ax.set_xlabel("")
    ax.set_ylim(0, pv.to_numpy().max() * 1.18)
    ax.set_title("Expanding-Window Temporal Cross-Validation on Dev Period (PR-AUC)")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(loc="upper left", frameon=False, title=None)
    save(fig, "fig6_cv_pr_auc.png")

# ---------------------------------------------------------------- Fig 7: Part 5 Cost threshold curve
df = read_csv("part5_threshold_curve.csv")
if df is not None:
    i = df["total_cost_USD_M"].idxmin()
    tau = df.loc[i, "threshold"]
    min_cost = df.loc[i, "total_cost_USD_M"]
    fig, ax = plt.subplots(1, 2, figsize=(10.0, 3.2))
    ax[0].plot(df["threshold"], df["total_cost_USD_M"], color=C_BLUE, lw=1.8)
    ax[0].axvline(tau, color=C_RED, ls="--", lw=1.2)
    ax[0].annotate(f"Locked tau* = {tau:.3f}\n(${min_cost:.1f}M)", (tau, min_cost),
                   xytext=(tau + 0.11, min_cost + (df["total_cost_USD_M"].max() - min_cost) * 0.30),
                   textcoords="data", color=C_RED, fontweight="bold", fontsize=8.8,
                   arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.0))
    ax[0].set_xlabel("Probability alarm threshold (tau)")
    ax[0].set_ylabel("Total cost on Dev OOF (USD M)")
    ax[0].set_title("(a) Cost-minimising threshold on Dev OOF")
    ax[1].plot(df["threshold"], df["recall"] * 100, label="Row Recall (%)", color=C_GREEN, lw=1.8)
    ax[1].plot(df["threshold"], df["precision"] * 100, label="Precision (%)", color=C_ORANGE, lw=1.8)
    ax[1].axvline(tau, color=C_RED, ls="--", lw=1.2, label=f"tau* = {tau:.2f}")
    ax[1].set_xlabel("Probability alarm threshold (tau)")
    ax[1].set_ylabel("Percentage (%)")
    ax[1].set_title("(b) Recall vs. Precision trade-off (Dev OOF)")
    ax[1].legend(loc="center right", frameon=False, fontsize=8.5)
    save(fig, "fig7_threshold_cost.png")

# ---------------------------------------------------------------- Fig 8: Part 5 Feature importance
df = read_csv("part5_feature_importance.csv")
if df is not None:
    top = df.sort_values("pr_auc_drop_mean", ascending=False).head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    ax.barh(top["feature"], top["pr_auc_drop_mean"], xerr=top["pr_auc_drop_std"], color=C_NAVY, height=0.65,
            error_kw=dict(ecolor=C_GREY, lw=1))
    for idx_bar, (val, std) in enumerate(zip(top["pr_auc_drop_mean"], top["pr_auc_drop_std"])):
        ax.text(val + std + 0.002, idx_bar, f"-{val:.4f}", va="center", fontsize=8, color="#1E293B")
    ax.set_xlim(0, top["pr_auc_drop_mean"].max() * 1.22)
    ax.set_xlabel("Mean PR-AUC drop when feature is permuted (Dev Fold-3: 2008Q1-2009Q4)")
    ax.set_title("Top 12 Features by Permutation Importance (HistGBDT on Dev Fold-3)")
    save(fig, "fig8_feature_importance.png")

# ---------------------------------------------------------------- Fig 9: Part 5 Single Test evaluation
df = read_csv("part5_test_evaluation.csv")
if df is not None:
    df["tau_num"] = pd.to_numeric(df["tau"], errors="coerce")
    mask_lock = df["threshold_rule"].astype(str).str.lower().str.contains("lock") | (df["tau_num"] < 0.25)
    d_lock = df[mask_lock].drop_duplicates(subset=["model"]).set_index("model")
    d_def = df[~mask_lock].drop_duplicates(subset=["model"]).set_index("model").reindex(d_lock.index)
    cost_lock = pd.to_numeric(d_lock["Test_Cost_USD_M"], errors="coerce")
    cost_def = pd.to_numeric(d_def["Test_Cost_USD_M"], errors="coerce")
    no_alarm = float(pd.to_numeric(d_lock["NoAlarm_Cost_USD_M"], errors="coerce").iloc[0])

    fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.2))
    x = np.arange(len(d_lock))
    bw = 0.35
    ax[0].bar(x - bw / 2, cost_def, bw, color=C_GREY, alpha=0.8, label="Default tau = 0.50")
    ax[0].bar(x + bw / 2, cost_lock, bw, color=C_NAVY, label="Dev-locked tau* = 0.05")
    ax[0].axhline(no_alarm, color=C_RED, ls="--", lw=1.2, label=f"No-alarm (${no_alarm:.1f}M)")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(d_lock.index)
    ax[0].set_ylim(0, no_alarm * 1.22)
    ax[0].set_ylabel("Total Test Cost (USD M)")
    ax[0].set_title("(a) Test Cost: Locked tau*=0.05 vs. Default 0.50")
    ax[0].legend(loc="upper right", fontsize=7.8, frameon=True, facecolor="white", framealpha=0.9)

    caught_def = pd.to_numeric(d_def["Banks_caught_of_199"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    caught_lock = pd.to_numeric(d_lock["Banks_caught_of_199"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    ax[1].bar(x - bw / 2, caught_def, bw, color=C_GREY, alpha=0.8, label="Default tau = 0.50")
    ax[1].bar(x + bw / 2, caught_lock, bw, color=C_GREEN, label="Dev-locked tau* = 0.05")
    ax[1].axhline(199, color=C_RED, ls=":", lw=1.1, label="Total failed banks (199)")
    for idx_x, val in enumerate(caught_lock):
        v = float(val)
        ax[1].text(idx_x + bw / 2, v + 4, f"{int(v)}/199\n({v/199*100:.1f}%)", ha="center", fontsize=7.8, fontweight="bold")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(d_lock.index)
    ax[1].set_ylim(0, 238)
    ax[1].set_ylabel("Failed banks caught (>=1 alarm in 4 qtrs)")
    ax[1].set_title("(b) Failed Banks Caught on Locked Test (2011Q1-2025Q2)")
    ax[1].legend(loc="lower right", fontsize=7.8, frameon=True, facecolor="white", framealpha=0.9)
    save(fig, "fig9_test_evaluation.png")

print("done")
