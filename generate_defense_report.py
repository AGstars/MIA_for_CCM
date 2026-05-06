from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


BASELINE_BEST: dict[str, dict] = {}


@dataclass(frozen=True)
class PlotGroup:
    title: str
    defense: str
    visibility: str


def _fmt(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    try:
        f = float(v)
        return f"{f:.4f}"
    except Exception:
        return str(v)


def _md_table(df: pd.DataFrame, cols: list[str]) -> str:
    view = df[cols].copy()
    for c in cols:
        if c not in ("defense", "visibility", "attack_model"):
            view[c] = view[c].map(_fmt)

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, r in view.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def _rel_plot_path(defense: str, visibility: str, filename: str) -> str:
    return f"./{defense}/plots/{visibility}/{filename}".replace("\\", "/")


def _img(defense: str, visibility: str, filename: str) -> str:
    return f"![]({_rel_plot_path(defense, visibility, filename)})"


def _existing(files: Iterable[str], runs_root: str) -> list[str]:
    out = []
    for f in files:
        if os.path.exists(os.path.join(runs_root, f)):
            out.append(f)
    return out


def _make_panel(runs_root: str, defense: str, visibility: str) -> str | None:
    base = os.path.join(runs_root, defense, "plots", visibility)
    paths = {
        "a": os.path.join(base, "metric_summary.png"),
        "b": os.path.join(base, "roc_curve_log.png"),
        "c": os.path.join(base, "det_curve_loglog.png"),
        "d": os.path.join(base, "tpr_bar_chart.png"),
    }
    if not all(os.path.exists(p) for p in paths.values()):
        return None

    imgs = {k: mpimg.imread(p) for k, p in paths.items()}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    order = ["a", "b", "c", "d"]
    for ax, key in zip(axes.flatten(), order, strict=False):
        ax.imshow(imgs[key])
        ax.axis("off")
        ax.text(
            0.02,
            0.98,
            f"({key})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            fontweight="bold",
            color="black",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 2},
        )

    fig.tight_layout()
    out_path = os.path.join(base, "panel_abcd.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _best_attack_from_metrics(metrics_csv: str) -> dict | None:
    if not os.path.exists(metrics_csv):
        return None
    df = pd.read_csv(metrics_csv)
    if df.empty:
        return None
    for c in ("AUC", "F1", "TPR@1%FPR", "TPR@0.1%FPR"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    key = "TPR@0.1%FPR" if "TPR@0.1%FPR" in df.columns else "AUC"
    sort_cols = [key]
    if "AUC" in df.columns and key != "AUC":
        sort_cols.append("AUC")
    r = df.sort_values(sort_cols, ascending=[False] * len(sort_cols)).head(1).iloc[0]
    return {
        "model": r.get("model"),
        "AUC": r.get("AUC"),
        "F1": r.get("F1"),
        "TPR@1%FPR": r.get("TPR@1%FPR"),
        "TPR@0.1%FPR": r.get("TPR@0.1%FPR"),
    }


def _baseline_best_map(runs_root: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    base_dir = os.path.join(runs_root, "baseline", "plots")
    if not os.path.isdir(base_dir):
        return out
    for vis in os.listdir(base_dir):
        metrics_csv = os.path.join(base_dir, vis, "metrics_table.csv")
        best = _best_attack_from_metrics(metrics_csv)
        if best is not None:
            out[vis] = best
    return out


def _plot_block(runs_root: str, g: PlotGroup) -> str:
    panel_path = _make_panel(runs_root, g.defense, g.visibility)
    if panel_path is None:
        return ""

    rel_panel = os.path.relpath(panel_path, runs_root).replace("\\", "/")
    metrics_csv = os.path.join(runs_root, g.defense, "plots", g.visibility, "metrics_table.csv")
    best = _best_attack_from_metrics(metrics_csv)
    best_line = ""
    if best is not None:
        best_line = (
            f"- 该场景下最强攻击器（按 `TPR@0.1%FPR`）：`{best['model']}`，"
            f"AUC={_fmt(best['AUC'])}，TPR@1%FPR={_fmt(best['TPR@1%FPR'])}，TPR@0.1%FPR={_fmt(best['TPR@0.1%FPR'])}"
        )

    summary_line = ""
    interpretation = ""
    ref = None
    ref_name = ""
    if g.defense == "baseline":
        if g.visibility != "full":
            ref = BASELINE_BEST.get("full")
            ref_name = "baseline(full)"
    else:
        ref = BASELINE_BEST.get(g.visibility) or BASELINE_BEST.get("full")
        ref_name = f"baseline({g.visibility if g.visibility in BASELINE_BEST else 'full'})"

    if best is not None and ref is not None:
        try:
            cur = float(best.get("TPR@0.1%FPR"))
            base = float(ref.get("TPR@0.1%FPR"))
            if base > 0:
                rel = (cur - base) / base
                sign = "+" if rel >= 0 else ""
                summary_line = (
                    f"- 结论：相对 {ref_name}，TPR@0.1%FPR 从 {_fmt(base)} 变为 {_fmt(cur)}（{sign}{rel*100:.1f}%）。"
                )

                if g.visibility in ("no_shadow", "loss_only", "prob_only"):
                    if rel < 0:
                        interpretation = "- 说明：接口/特征可见性限制会显著削弱穿透，攻击高度依赖可见的概率与影子校准信号。"
                    else:
                        interpretation = "- 说明：在该限制下仍残留一定可分性，可能来自目标模型的损失/困惑度差异。"
                elif g.defense != "baseline":
                    if rel < 0:
                        interpretation = "- 说明：该训练防御对成员推断具有一定抑制效果（低误报区间穿透下降）。"
                    else:
                        interpretation = "- 说明：该训练防御未能抑制成员推断，甚至可能强化了可分性（需进一步检查过拟合程度/训练轮次/早停触发）。"
        except Exception:
            summary_line = ""
            interpretation = ""

    explain = "\n".join(
        [
            "- (a) `metric_summary`：汇总 AUC/F1/TPR@低FPR，直观看穿透强度",
            "- (b) `roc_curve_log`：log-scale ROC，强调低 FPR 区域的可分性",
            "- (c) `det_curve_loglog`：DET(log-log)，越靠左下越好，强调低误报下的漏报",
            "- (d) `tpr_bar_chart`：对比不同攻击器在低 FPR 下的 TPR",
        ]
    )

    lines = [f"### {g.title}", "", f"![](./{rel_panel})", "", explain]
    if best_line:
        lines.extend(["", best_line])
    if summary_line:
        lines.extend(["", summary_line])
    if interpretation:
        lines.extend(["", interpretation])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--runs-root", default="defense_runs")
    p.add_argument("--summary-csv", default="")
    p.add_argument("--output", default="")
    args = p.parse_args(argv)

    runs_root = args.runs_root

    global BASELINE_BEST
    BASELINE_BEST = _baseline_best_map(runs_root)
    summary_csv = args.summary_csv or os.path.join(runs_root, "defense_sweep_summary.csv")
    out_path = args.output or os.path.join(runs_root, "DEFENSE_SWEEP_REPORT.md")

    df = pd.read_csv(summary_csv)
    for c in ("auc", "f1", "tpr_at_1pct_fpr", "tpr_at_0p1pct_fpr"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    defenses = sorted(df["defense"].dropna().unique().tolist())
    vis = sorted(df["visibility"].dropna().unique().tolist())
    models = sorted(df["attack_model"].dropna().unique().tolist())

    full = df[df["visibility"] == "full"].copy()
    best_full = (
        full.sort_values(["defense", "tpr_at_0p1pct_fpr", "auc"], ascending=[True, False, False])
        .groupby(["defense"], as_index=False)
        .head(1)
    )

    baseline_full = df[(df["defense"] == "baseline") & (df["visibility"] == "full")].copy()
    baseline_best = None
    if not baseline_full.empty:
        baseline_best = baseline_full.sort_values(["tpr_at_0p1pct_fpr", "auc"], ascending=[False, False]).head(1).iloc[0]

    full_best_attack = None
    full_worst_attack = None
    if not best_full.empty:
        full_best_attack = best_full.sort_values(["tpr_at_0p1pct_fpr", "auc"], ascending=[False, False]).head(1).iloc[0]
        full_worst_attack = best_full.sort_values(["tpr_at_0p1pct_fpr", "auc"], ascending=[True, True]).head(1).iloc[0]

    lines: list[str] = []
    lines.append("# Defense Sweep 结果总结")
    lines.append("")
    lines.append(f"结果目录：`{runs_root}`")
    lines.append(f"汇总表：`{os.path.relpath(summary_csv, runs_root) if os.path.isabs(summary_csv) else os.path.basename(summary_csv)}`")
    lines.append("")

    lines.append("## 实验配置概览")
    lines.append("")
    lines.append(f"- 防御配置（{len(defenses)} 个）：" + ", ".join(f"`{d}`" for d in defenses))
    lines.append(f"- 可见性场景（{len(vis)} 个）：" + ", ".join(f"`{v}`" for v in vis))
    lines.append(f"- 攻击器（{len(models)} 个）：" + ", ".join(f"`{m}`" for m in models))
    lines.append("")

    lines.append("## 核心指标与解读")
    lines.append("")
    lines.append("- `TPR@0.1%FPR`：最严苛的低误报指标，最能体现“高置信成员推断”下的穿透程度")
    lines.append("- `TPR@1%FPR`：相对宽松但仍偏安全侧的指标")
    lines.append("- `AUC`：整体区分能力（不固定阈值）")
    lines.append("")

    lines.append("## Full 可见性下的穿透强度（按 TPR@0.1%FPR 取每个防御下最强攻击器）")
    lines.append("")
    if not best_full.empty:
        lines.append(
            _md_table(
                best_full,
                ["defense", "attack_model", "auc", "tpr_at_1pct_fpr", "tpr_at_0p1pct_fpr"],
            )
        )
        lines.append("")

    if baseline_best is not None:
        lines.append("## Baseline (full) 参考")
        lines.append("")
        lines.append(
            f"- Baseline(full) 最强攻击器（按 `TPR@0.1%FPR`）：`{baseline_best['attack_model']}`，"
            f"AUC={_fmt(baseline_best['auc'])}，TPR@1%FPR={_fmt(baseline_best['tpr_at_1pct_fpr'])}，TPR@0.1%FPR={_fmt(baseline_best['tpr_at_0p1pct_fpr'])}"
        )
        lines.append("")

    if full_best_attack is not None and full_worst_attack is not None:
        lines.append("## Full 下穿透强弱对比")
        lines.append("")
        lines.append(
            f"- full 场景下穿透最强：`defense={full_best_attack['defense']}` + `attack={full_best_attack['attack_model']}`，"
            f"TPR@0.1%FPR={_fmt(full_best_attack['tpr_at_0p1pct_fpr'])}，AUC={_fmt(full_best_attack['auc'])}"
        )
        lines.append(
            f"- full 场景下穿透最弱：`defense={full_worst_attack['defense']}` + `attack={full_worst_attack['attack_model']}`，"
            f"TPR@0.1%FPR={_fmt(full_worst_attack['tpr_at_0p1pct_fpr'])}，AUC={_fmt(full_worst_attack['auc'])}"
        )
        lines.append("")

    lines.append("## Full 场景下各攻击器明细")
    lines.append("")
    full_detail = df[df["visibility"] == "full"].sort_values(["defense", "attack_model"]).copy()
    lines.append(
        _md_table(
            full_detail,
            ["defense", "attack_model", "auc", "tpr_at_1pct_fpr", "tpr_at_0p1pct_fpr"],
        )
    )
    lines.append("")

    lines.append("## 关键场景图表")
    lines.append("")
    lines.append("每个场景默认展示 4 张：`metric_summary / roc_curve_log / det_curve_loglog / tpr_bar_chart`（若缺失则自动补齐其他可用图）。")
    lines.append("")

    groups = [
        PlotGroup(title="Baseline / full", defense="baseline", visibility="full"),
        PlotGroup(title="Baseline / no_shadow", defense="baseline", visibility="no_shadow"),
        PlotGroup(title="Baseline / loss_only", defense="baseline", visibility="loss_only"),
        PlotGroup(title="Combo defense (combo_reg) / full", defense="combo_reg", visibility="full"),
    ]
    for g in groups:
        block = _plot_block(runs_root, g)
        if block:
            lines.append(block)

    lines.append("## 复现定位")
    lines.append("")
    lines.append("- 每个防御目录：`<defense>/`")
    lines.append("- 特征 CSV：`<defense>/attack_train_dataset.csv` 与 `<defense>/attack_test_dataset.csv`")
    lines.append("- 图表与每场景指标：`<defense>/plots/<visibility>/`（含 `metrics_table.csv` 与 PNG 图）")
    lines.append("")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")

    print(f"✅ Report written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
