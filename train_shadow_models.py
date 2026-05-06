import argparse
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 关闭所有警告 + 解决 libgomp 错误
import warnings
warnings.filterwarnings('ignore')
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments
)


def _maybe_set_dropout(model, dropout: float) -> None:
    if dropout is None:
        return
    try:
        d = float(dropout)
    except Exception:
        return
    if d < 0.0:
        d = 0.0
    if d > 1.0:
        d = 1.0
    cfg = getattr(model, "config", None)
    if cfg is None:
        return
    for k in ("attn_pdrop", "resid_pdrop", "embd_pdrop"):
        if hasattr(cfg, k):
            setattr(cfg, k, d)


def _has_saved_model(output_dir: str) -> bool:
    if not os.path.isdir(output_dir):
        return False
    for fname in ("model.safetensors", "pytorch_model.bin", "config.json"):
        if os.path.exists(os.path.join(output_dir, fname)):
            return True
    return False

def main(argv: list[str] | None = None):
    print("=== 训练“模拟者”——影子模型 (Shadow Models) ===")

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--num-shadow-models", type=int, default=int(os.environ.get("NUM_SHADOW_MODELS", "5")))
    parser.add_argument("--target-train-size", type=int, default=int(os.environ.get("TARGET_TRAIN_SIZE", "0")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SHADOW_SEED", "42")))
    parser.add_argument("--data-dir", default="mia_data")
    parser.add_argument("--model-name", default="microsoft/CodeGPT-small-java-adaptedGPT2")
    parser.add_argument("--output-root", default="models")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=2)
    args = parser.parse_args(argv)
    
    # 1. 检查硬件并设定路径
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"正在使用设备: {device}")
    
    data_dir = args.data_dir
    
    # 2. 加载处理好的 Shadow 数据集
    print("\n加载第二步处理好的 shadow_train 和 shadow_test 数据集...")
    try:
        shadow_train = load_from_disk(os.path.join(data_dir, "shadow_train"))
        shadow_test = load_from_disk(os.path.join(data_dir, "shadow_test"))
        print(f"加载成功！Train 数量: {len(shadow_train)}, Test 数量: {len(shadow_test)}")
    except Exception as e:
        print(f"加载数据集失败，请确保第二步数据准备已完成。错误: {e}")
        return

    if int(args.target_train_size) > 0:
        target_train_size = int(args.target_train_size)
    else:
        try:
            target_train = load_from_disk(os.path.join(data_dir, "target_train"))
            target_train_size = int(len(target_train))
        except Exception:
            target_train_size = 8000

    # 3. 加载 Tokenizer 和 DataCollator
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    num_shadow_models = int(args.num_shadow_models)
    print(f"\n计划训练 {num_shadow_models} 个独立的影子模型...")

    start_index = int(args.start_index)
    if start_index < 0:
        start_index = 0
    if start_index >= num_shadow_models:
        print(f"start-index={start_index} 超过 num-shadow-models={num_shadow_models}，无需训练。")
        return

    for i in range(start_index, num_shadow_models):
        print(f"\n==============================================")
        print(f"🚀 正在训练影子模型 [{i+1}/{num_shadow_models}]")
        print(f"==============================================")
        
        output_dir = os.path.join(args.output_root, f"shadow_model_{i}")
        if bool(args.skip_existing) and _has_saved_model(output_dir):
            print(f"⏭️  已存在模型，跳过：{output_dir}")
            continue
        os.makedirs(output_dir, exist_ok=True)
        
        current_shadow_train = shadow_train.shuffle(seed=int(args.seed) + i).select(range(target_train_size))
        print(f"当前影子模型专属训练集大小: {len(current_shadow_train)} (与 Target 模型保持一致)")
        
        # 初始化新模型
        model = AutoModelForCausalLM.from_pretrained(model_name)
        _maybe_set_dropout(model, args.dropout)
        
        use_early_stop = int(args.early_stopping_patience) > 0
        save_strategy = "epoch" if use_early_stop else "no"
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=int(args.epochs),
            per_device_train_batch_size=int(args.train_batch_size),
            per_device_eval_batch_size=int(args.eval_batch_size),
            gradient_accumulation_steps=int(args.grad_accum),
            evaluation_strategy="epoch",
            save_strategy=save_strategy,
            save_total_limit=1 if use_early_stop else None,
            logging_steps=50,
            learning_rate=float(args.lr),
            weight_decay=float(args.weight_decay),
            warmup_ratio=float(args.warmup_ratio),
            label_smoothing_factor=float(args.label_smoothing),
            bf16=True if device == "cuda" else False,
            dataloader_num_workers=4,
            load_best_model_at_end=True if use_early_stop else False,
            metric_for_best_model="eval_loss" if use_early_stop else None,
            greater_is_better=False if use_early_stop else None,
            save_only_model=True,
            seed=int(args.seed) + i,
        )

        # 训练器
        callbacks = [EarlyStoppingCallback(early_stopping_patience=int(args.early_stopping_patience))] if use_early_stop else None
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=current_shadow_train,
            eval_dataset=shadow_test,
            data_collator=data_collator,
            callbacks=callbacks,
        )

        # 开始训练
        trainer.train()

        # 保存最终模型
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"🎉 影子模型 [{i+1}] 保存完成！")

    print("\n✅ 所有影子模型训练完成！")

if __name__ == "__main__":
    main()
