# AutoDL_MIA：代码补全模型的成员推理攻击（MIA）复现

本目录提供一个可复现的成员推理攻击（Membership Inference Attack, MIA）实验流水线，面向自回归式代码补全模型（Causal LM / CodeGPT 风格），评估其对训练样本的隐私泄漏风险。

覆盖能力：

- 目标模型与影子模型训练（Causal LM）
- 多维特征提取：统计特征（Loss/PPL/熵等）+ AST 结构特征 + LiRA 差异特征 + 句向量池化特征（Embedding）
- 攻击模型训练与评估：RandomForest / XGBoost / LogisticRegression + 阈值基线
- 低误报安全指标：TPR @ 1% FPR、TPR @ 0.1% FPR
- 论文图表输出：ROC（含 log-scale）、PR、TPR@lowFPR 柱状图、DET、阈值诊断曲线、混淆矩阵、特征重要性 Top-K
- 少样本与可解释评估：FeS 风格少样本攻击训练，并输出 Log-MIA 报告（逐样本分数 + 低 FPR 校准阈值）

更详细的分步复现与常见问题见：[用户指导复现手册](./用户指导复现手册.md)

## 快速开始

在 Linux GPU 环境（推荐 AutoDL / Ubuntu 22.04 + CUDA）进入 `AutoDL_MIA/`：

```bash
pip install -r requirements.txt
bash run_all.sh
```

少样本模式（资源受限 / 快速隐私体检）：

```bash
MIA_MODE=fes bash run_all.sh
```

说明：当 `NUM_SHADOW_MODELS=1` 时，LiRA 相关特征会退化（参考模型为空），本实现会把 LiRA 差异特征置为 0 以避免 NaN；若希望 LiRA 有效，建议设置 `NUM_SHADOW_MODELS>=2`。

## 关键脚本

- `prepare_mia_data.py`：下载 Concode 并严格划分 Target/Shadow 数据集
- `train_target_model.py`：训练目标模型（Causal LM）
- `train_shadow_models.py`：训练多个影子模型（Causal LM）
- `extract_features.py`：提取攻击特征（统计 + AST + Embedding + LiRA）
- `mia_pipeline/cli.py`：攻击评估入口（classic / fes）与 Log-MIA 报告输出
- `run_all.sh`：一键流水线（支持跳过阶段）

## 输出产物

- `mia_data/`、`mia_data_raw/`：数据集缓存
- `models/target_model/`、`models/shadow_model_*/`：训练好的模型
- `attack_train_dataset.csv`、`attack_test_dataset.csv`：攻击训练/测试特征表
- `plots/main_models/`：主实验图表与 `metrics_table.csv`
- `plots/ablation/`：消融图表（如你使用对应脚本生成）
- `plots/main_models/logmia_*.{csv,json,md}`：Log-MIA 报告（少样本模式默认生成）

## GitHub 备注

仓库根目录已提供 `.gitignore`，默认忽略训练产物（`models/`、`mia_data*/`、`plots/` 等）与中间 CSV，避免误把大文件提交到 GitHub。

## 复现开关（run_all.sh）

阶段跳过（值为 `1` 表示跳过）：

- `SKIP_INSTALL`：跳过 `pip install`
- `SKIP_DATA`：跳过数据准备
- `SKIP_TRAIN_TARGET`：跳过目标模型训练
- `SKIP_TRAIN_SHADOW`：跳过影子模型训练
- `SKIP_EXTRACT`：跳过特征提取
- `SKIP_ATTACK`：跳过攻击评估与画图

示例：特征 CSV 已存在，仅重跑攻击评估：

```bash
SKIP_INSTALL=1 SKIP_DATA=1 SKIP_TRAIN_TARGET=1 SKIP_TRAIN_SHADOW=1 SKIP_EXTRACT=1 bash run_all.sh
```

示例：少样本模式下仅重跑攻击评估并生成 Log-MIA：

```bash
SKIP_INSTALL=1 SKIP_DATA=1 SKIP_TRAIN_TARGET=1 SKIP_TRAIN_SHADOW=1 SKIP_EXTRACT=1 MIA_MODE=fes bash run_all.sh
```

## 常见问题

- `RuntimeError: xgboost is not installed but requested`：安装 `xgboost` 或把 `ATTACK_MODELS` 中的 `xgb` 删除，例如 `ATTACK_MODELS=rf,thr_loss`。
- 国内网络下载慢/失败：脚本默认设置 `HF_ENDPOINT=https://hf-mirror.com`；如你可直连 HuggingFace，可在脚本中移除该行。

## 引用

如本仓库对你的研究有帮助，欢迎在论文/报告中引用（你可以在上传 GitHub 后补充 BibTeX）。
