from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFENSE_ORDER = [
    "baseline",
    "wd_0p01",
    "dropout_0p1",
    "labelsmooth_0p1",
    "earlystop_p3",
    "combo_reg",
]

VIS_ORDER = [
    "full",
    "no_ast",
    "no_shadow",
    "loss_only",
    "prob_only",
]

MODEL_ORDER = [
    "XGBoost (Full)",
    "Random Forest",
    "Threshold: Target Loss",
]


def _read_summary(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ("tpr_at_1pct_fpr", "tpr_at_0p1pct_fpr"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _order(values: list[str], preferred: list[str]) -> list[str]:
    s = set(values)
    out = [v for v in preferred if v in s]
    out.extend([v for v in values if v not in set(out)])
    return out


def build_figure(df: pd.DataFrame, defenses: list[str], vis: list[str], models: list[str]) -> plt.Figure:
    nrows = len(defenses)
    ncols = len(vis)
    fig_w = max(14, 3.2 * ncols)
    fig_h = max(10, 1.9 * nrows)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), sharey=True)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[a] for a in axes])

    width = 0.35
    x = np.arange(len(models))

    for i, d in enumerate(defenses):
        for j, v in enumerate(vis):
            ax = axes[i, j]
            sub = df[(df["defense"] == d) & (df["visibility"] == v)].copy()
            sub = sub.set_index("attack_model").reindex(models)

            y1 = sub["tpr_at_1pct_fpr"].to_numpy()
            y01 = sub["tpr_at_0p1pct_fpr"].to_numpy()

            ax.bar(x - width / 2, y1, width, label="TPR@1%FPR", color="#4C78A8")
            ax.bar(x + width / 2, y01, width, label="TPR@0.1%FPR", color="#F58518")

            ax.set_ylim(0.0, float(np.nanmax(df[["tpr_at_1pct_fpr", "tpr_at_0p1pct_fpr"]].to_numpy())) * 1.15)
            ax.grid(axis="y", alpha=0.25)

            if i == 0:
                ax.set_title(v, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{d}\nTPR", fontsize=9)
            if i == nrows - 1:
                ax.set_xticks(x)
                ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
            else:
                ax.set_xticks(x)
                ax.set_xticklabels([])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Defense Sweep: TPR@lowFPR comparison across defenses & visibility", y=0.998, fontsize=12)
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.975))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--runs-root", default="defense_runs")
    p.add_argument("--summary-csv", default="")
    p.add_argument("--output", default="")
    args = p.parse_args(argv)

    runs_root = args.runs_root
    summary_csv = args.summary_csv or os.path.join(runs_root, "defense_sweep_summary.csv")
    if not os.path.exists(summary_csv):
        raise FileNotFoundError(f"summary csv not found: {summary_csv}")

    df = _read_summary(summary_csv)
    defenses = _order(df["defense"].dropna().unique().tolist(), DEFENSE_ORDER)
    vis = _order(df["visibility"].dropna().unique().tolist(), VIS_ORDER)
    models = _order(df["attack_model"].dropna().unique().tolist(), MODEL_ORDER)

    fig = build_figure(df, defenses, vis, models)
    out_path = args.output or os.path.join(runs_root, "tpr_bar_chart_all.png")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
