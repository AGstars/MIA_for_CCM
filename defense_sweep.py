from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import pandas as pd

from extract_features import main as extract_main
from mia_pipeline.cli import build_parser as build_eval_parser
from mia_pipeline.cli import run as eval_run
from train_shadow_models import main as train_shadow_main
from train_target_model import main as train_target_main


@dataclass(frozen=True)
class DefenseConfig:
    name: str
    epochs: int = 20
    lr: float = 3e-5
    weight_decay: float = 0.0
    dropout: float = 0.0
    label_smoothing: float = 0.0
    early_stopping_patience: int = 0


@dataclass(frozen=True)
class VisibilityScenario:
    name: str
    ignore_contains: str = ""
    only_contains: str = ""


PRESET_DEFENSES: list[DefenseConfig] = [
    DefenseConfig(name="baseline"),
    DefenseConfig(name="wd_0p01", weight_decay=0.01),
    DefenseConfig(name="dropout_0p1", dropout=0.1),
    DefenseConfig(name="labelsmooth_0p1", label_smoothing=0.1),
    DefenseConfig(name="earlystop_p3", early_stopping_patience=3, epochs=50),
    DefenseConfig(name="combo_reg", weight_decay=0.01, dropout=0.1, label_smoothing=0.1, early_stopping_patience=3, epochs=50),
]


PRESET_VIS: list[VisibilityScenario] = [
    VisibilityScenario(name="full"),
    VisibilityScenario(name="no_ast", ignore_contains="ast_"),
    VisibilityScenario(name="no_shadow", ignore_contains="ref_,lira_"),
    VisibilityScenario(name="loss_only", only_contains="eval_loss_mean,eval_perplexity"),
    VisibilityScenario(name="prob_only", only_contains="eval_max_prob_mean,eval_entropy_mean"),
]


def _iter_defenses(names: str) -> list[DefenseConfig]:
    wanted = [n.strip() for n in names.split(",") if n.strip()]
    if not wanted or wanted == ["all"]:
        return PRESET_DEFENSES
    table = {d.name: d for d in PRESET_DEFENSES}
    out: list[DefenseConfig] = []
    for n in wanted:
        if n not in table:
            raise ValueError(f"unknown defense preset: {n}")
        out.append(table[n])
    return out


def _iter_visibility(names: str) -> list[VisibilityScenario]:
    wanted = [n.strip() for n in names.split(",") if n.strip()]
    if not wanted or wanted == ["all"]:
        return PRESET_VIS
    table = {v.name: v for v in PRESET_VIS}
    out: list[VisibilityScenario] = []
    for n in wanted:
        if n not in table:
            raise ValueError(f"unknown visibility preset: {n}")
        out.append(table[n])
    return out


def _train_models(run_dir: str, data_dir: str, num_shadow_models: int, defense: DefenseConfig) -> tuple[str, str]:
    model_root = os.path.join(run_dir, "models")
    target_dir = os.path.join(model_root, "target_model")
    os.makedirs(model_root, exist_ok=True)

    train_target_main(
        [
            "--data-dir",
            data_dir,
            "--output-dir",
            target_dir,
            "--skip-existing",
            "--epochs",
            str(defense.epochs),
            "--lr",
            str(defense.lr),
            "--weight-decay",
            str(defense.weight_decay),
            "--dropout",
            str(defense.dropout),
            "--label-smoothing",
            str(defense.label_smoothing),
            "--early-stopping-patience",
            str(defense.early_stopping_patience),
        ]
    )
    train_shadow_main(
        [
            "--data-dir",
            data_dir,
            "--output-root",
            model_root,
            "--num-shadow-models",
            str(num_shadow_models),
            "--skip-existing",
            "--epochs",
            str(defense.epochs),
            "--lr",
            str(defense.lr),
            "--weight-decay",
            str(defense.weight_decay),
            "--dropout",
            str(defense.dropout),
            "--label-smoothing",
            str(defense.label_smoothing),
            "--early-stopping-patience",
            str(defense.early_stopping_patience),
        ]
    )
    return model_root, target_dir


def _extract_features(
    run_dir: str,
    data_dir: str,
    raw_data_dir: str,
    model_root: str,
    target_dir: str,
    num_shadow_models: int,
    debug: bool,
    debug_sample_num: int,
) -> tuple[str, str]:
    train_csv = os.path.join(run_dir, "attack_train_dataset.csv")
    test_csv = os.path.join(run_dir, "attack_test_dataset.csv")
    extract_args = [
        "--data-dir",
        data_dir,
        "--raw-data-dir",
        raw_data_dir,
        "--target-model-dir",
        target_dir,
        "--shadow-model-root",
        model_root,
        "--num-shadow-models",
        str(num_shadow_models),
        "--output-dir",
        run_dir,
        "--output-train-csv",
        train_csv,
        "--output-test-csv",
        test_csv,
    ]
    if debug:
        extract_args.extend(["--debug", "--debug-sample-num", str(debug_sample_num)])
    extract_main(extract_args)
    return train_csv, test_csv


def _evaluate_one(
    train_csv: str,
    test_csv: str,
    out_dir: str,
    models: str,
    ignore_contains: str,
    only_contains: str,
    ignore_weak_ast: bool,
    n_jobs: int,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    parser = build_eval_parser()
    args = parser.parse_args(
        [
            "--train-csv",
            train_csv,
            "--test-csv",
            test_csv,
            "--output-dir",
            out_dir,
            "--models",
            models,
            "--n-jobs",
            str(n_jobs),
            "--ignore-contains",
            ignore_contains,
            "--only-contains",
            only_contains,
        ]
        + (["--ignore-weak-ast"] if ignore_weak_ast else [])
    )
    eval_run(args)
    return os.path.join(out_dir, "metrics_table.csv")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--runs-root", default="defense_runs")
    p.add_argument("--defenses", default="all")
    p.add_argument("--visibility", default="all")
    p.add_argument("--models", default="rf,xgb,thr_loss")
    p.add_argument("--data-dir", default="mia_data")
    p.add_argument("--raw-data-dir", default="mia_data_raw")
    p.add_argument("--num-shadow-models", type=int, default=int(os.environ.get("NUM_SHADOW_MODELS", "5")))
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-extract", action="store_true")
    p.add_argument("--existing-model-root", default="")
    p.add_argument("--existing-target-dir", default="")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--debug-sample-num", type=int, default=40)
    p.add_argument("--ignore-weak-ast", action="store_true")
    p.add_argument("--n-jobs", type=int, default=16)
    args = p.parse_args(argv)

    defenses = _iter_defenses(args.defenses)
    scenarios = _iter_visibility(args.visibility)
    os.makedirs(args.runs_root, exist_ok=True)

    rows: list[dict] = []
    for d in defenses:
        run_dir = os.path.join(args.runs_root, d.name)
        os.makedirs(run_dir, exist_ok=True)

        model_root = os.path.join(run_dir, "models")
        target_dir = os.path.join(model_root, "target_model")
        if not args.skip_train:
            model_root, target_dir = _train_models(run_dir, args.data_dir, int(args.num_shadow_models), d)
        else:
            if args.existing_model_root:
                model_root = args.existing_model_root
            if args.existing_target_dir:
                target_dir = args.existing_target_dir

        train_csv = os.path.join(run_dir, "attack_train_dataset.csv")
        test_csv = os.path.join(run_dir, "attack_test_dataset.csv")
        if bool(args.resume) and os.path.exists(train_csv) and os.path.exists(test_csv):
            pass
        elif not args.skip_extract:
            train_csv, test_csv = _extract_features(
                run_dir=run_dir,
                data_dir=args.data_dir,
                raw_data_dir=args.raw_data_dir,
                model_root=model_root,
                target_dir=target_dir,
                num_shadow_models=int(args.num_shadow_models),
                debug=bool(args.debug),
                debug_sample_num=int(args.debug_sample_num),
            )

        for s in scenarios:
            out_dir = os.path.join(run_dir, "plots", s.name)
            metrics_csv_existing = os.path.join(out_dir, "metrics_table.csv")
            if bool(args.resume) and os.path.exists(metrics_csv_existing):
                metrics_csv = metrics_csv_existing
            else:
                metrics_csv = _evaluate_one(
                    train_csv=train_csv,
                    test_csv=test_csv,
                    out_dir=out_dir,
                    models=args.models,
                    ignore_contains=s.ignore_contains,
                    only_contains=s.only_contains,
                    ignore_weak_ast=bool(args.ignore_weak_ast),
                    n_jobs=int(args.n_jobs),
                )
            df = pd.read_csv(metrics_csv)

            def _pick(row: pd.Series, keys: list[str]):
                for k in keys:
                    if k in row and pd.notna(row[k]):
                        return row[k]
                return None

            for _, r in df.iterrows():
                rows.append(
                    {
                        "defense": d.name,
                        "visibility": s.name,
                        "attack_model": r.get("model"),
                        "auc": _pick(r, ["auc", "AUC"]),
                        "f1": _pick(r, ["f1", "F1"]),
                        "tpr_at_1pct_fpr": _pick(r, ["tpr_at_1pct_fpr", "TPR@1%FPR"]),
                        "tpr_at_0p1pct_fpr": _pick(r, ["tpr_at_0p1pct_fpr", "TPR@0.1%FPR"]),
                    }
                )

    summary_path = os.path.join(args.runs_root, "defense_sweep_summary.csv")
    pd.DataFrame(rows).to_csv(summary_path, index=False, encoding="utf-8")
    print(f"\n✅ 防御穿透评估汇总已写入: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
