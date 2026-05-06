#!/bin/bash
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

echo "========================================================="
echo "🚀 正在初始化 AutoDL RTX 4090D (24GB) MIA 实验环境 🚀"
echo "========================================================="

if [ "${MIA_MODE:-classic}" = "fes" ]; then
  export TOTAL_SAMPLES="${TOTAL_SAMPLES:-30000}"
  export TARGET_TOTAL_SAMPLES="${TARGET_TOTAL_SAMPLES:-2000}"
  export SHADOW_START="${SHADOW_START:-10000}"
  export SHADOW_TOTAL_SAMPLES="${SHADOW_TOTAL_SAMPLES:-20000}"
  export NUM_SHADOW_MODELS="${NUM_SHADOW_MODELS:-1}"
fi

echo "=== 1/6 安装依赖 ==="
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
if [ "${SKIP_INSTALL:-0}" != "1" ]; then
  pip install -r requirements.txt
fi

echo "=== 2/6 准备数据 (CodeGPT Tokenizer & 缩减版 Target 数据) ==="
if [ "${SKIP_DATA:-0}" != "1" ]; then
  python prepare_mia_data.py
fi

echo "=== 3/6 训练目标模型 (Target Model - Causal LM) ==="
if [ "${SKIP_TRAIN_TARGET:-0}" != "1" ]; then
  python train_target_model.py
fi

echo "=== 4/6 训练影子模型池 (Shadow Models - Causal LM) ==="
if [ "${SKIP_TRAIN_SHADOW:-0}" != "1" ]; then
  python train_shadow_models.py
fi

echo "=== 5/6 提取攻击特征 (LiRA 似然比特征) ==="
if [ "${SKIP_EXTRACT:-0}" != "1" ]; then
  python extract_features.py
fi

echo "=== 6/6 训练并验证攻击分类器 ==="
if [ "${SKIP_ATTACK:-0}" != "1" ]; then
  if [ "${MIA_MODE:-classic}" = "fes" ]; then
    python -m mia_pipeline.cli \
      --mode fes \
      --shots-per-class "${FEWSHOT_SHOTS_PER_CLASS:-64}" \
      --models "${ATTACK_MODELS:-lr,thr_loss,rf}" \
      --output-dir "${ATTACK_OUTPUT_DIR:-plots/main_models}" \
      --ignore-weak-ast \
      --logmia
  else
    python -m mia_pipeline.cli \
      --mode classic \
      --models "${ATTACK_MODELS:-rf,xgb,thr_loss,thr_lira_z}" \
      --output-dir "${ATTACK_OUTPUT_DIR:-plots/main_models}" \
      --ignore-weak-ast
  fi
fi

echo "========================================================="
echo "🎉 所有任务执行完毕！请查看终端输出的 AUC 分数评估攻击效果 🎉"
echo "========================================================="
