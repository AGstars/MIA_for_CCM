# Repository Structure / 项目结构说明

This repository is organized to facilitate the reproduction of Membership Inference Attacks (MIA) on code completion models. 
To keep the repository lightweight, all large generated artifacts (models, datasets, plots) are ignored by Git.

## 📂 Directory Tree / 目录结构

```text
.
├── AutoDL_MIA/                 # 🚀 Main experimental pipeline (Recommended / 核心推荐入口)
│   ├── mia_pipeline/           # Attack evaluation & metrics module (评估模块)
│   │   ├── cli.py              # CLI entry point for evaluation
│   │   ├── models.py           # Classifier models (RF, XGB, LR, Threshold)
│   │   ├── plots.py            # Visualization generators
│   │   └── report.py           # Log-MIA report generator
│   ├── prepare_mia_data.py     # Data download and partition script
│   ├── train_target_model.py   # Target model training script
│   ├── train_shadow_models.py  # Shadow model training script
│   ├── extract_features.py     # Feature extraction (LiRA, AST, Embedding)
│   ├── defense_sweep.py        # Automated defense testing (可选的防御测试评估脚本)
│   ├── run_all.sh              # One-click execution script
│   ├── requirements.txt        # Python dependencies
│   ├── 用户指导复现手册.md         # Detailed reproduction guide (Chinese)
│   ├── README.md               # AutoDL_MIA Documentation (Chinese)
│   └── README_en.md            # AutoDL_MIA Documentation (English)
│
├── Core_code_ver1/             # 🗄️ Legacy code version 1 (Archive)
├── Core_code_ver2/             # 🗄️ Legacy code version 2 (Archive)
│
├── .gitignore                  # Git ignore rules
└── structure.md                # This structure documentation
```

## 🚫 Ignored Files / 被忽略的文件 (不会上传至 GitHub)

Based on the `.gitignore` rules, the following files are generated during execution but **will not be pushed** to GitHub to save bandwidth and storage:

1. **Models (`**/models/`)**: Target and shadow models (`.safetensors`, `.bin`).
2. **Datasets (`**/mia_data/`, `**/mia_data_raw/`)**: HuggingFace cached datasets (`.arrow`).
3. **Extracted Features (`*.csv`)**: `attack_train_dataset.csv`, `attack_test_dataset.csv`.
4. **Plots & Results (`**/plots/`)**: Generated PNGs, metrics tables, and evaluation logs (except `README.md` inside `plots`).
5. **Defense Runs (`**/defense_runs/`)**: Model checkpoints and output charts generated during `defense_sweep.py` evaluations.
6. **Environment & Caches**: `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`, `*.log`.

## 💡 How to navigate / 如何阅读源码

- **Entry Point**: Start with `AutoDL_MIA/run_all.sh` to understand the overall 6-step pipeline (from data prep to plotting).
- **Feature Extraction**: Check `AutoDL_MIA/extract_features.py` for how Multi-dimensional features (LiRA, AST, Embedding) are computed efficiently.
- **Evaluation Logic**: Look into `AutoDL_MIA/mia_pipeline/` for the implementation of the attack models, log-odds calculation, and low FPR evaluation metrics.
