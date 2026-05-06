import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
import argparse
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
    print("=== 训练“受害者”——目标模型 (Target Model) ===")

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--data-dir", default="mia_data")
    parser.add_argument("--output-dir", default="models/target_model")
    parser.add_argument("--model-name", default="microsoft/CodeGPT-small-java-adaptedGPT2")
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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    
    # 1. 检查硬件并设定路径
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"正在使用设备: {device}")
    
    data_dir = args.data_dir
    output_dir = args.output_dir
    if bool(args.skip_existing) and _has_saved_model(output_dir):
        print(f"⏭️  已存在目标模型，跳过训练：{output_dir}")
        return
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. 编写 DataLoader: 加载处理好的 Target 数据集
    print("\n加载第二步处理好的 target_train 和 target_test 数据集...")
    try:
        target_train = load_from_disk(os.path.join(data_dir, "target_train"))
        target_test = load_from_disk(os.path.join(data_dir, "target_test"))
        print(f"加载成功！Train 数量: {len(target_train)}, Test 数量: {len(target_test)}")
    except Exception as e:
        print(f"加载数据集失败，请确保第二步数据准备已完成。错误: {e}")
        return

    # 3. 加载基座模型和 Tokenizer
    model_name = args.model_name
    print(f"\n加载基座模型和 Tokenizer ({model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    _maybe_set_dropout(model, args.dropout)
    
    # 4. 实现 Causal LM 的 Data Collator
    print("配置 Data Collator (用于因果语言模型计算 Loss)...")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # 5. 定义训练循环参数 (Trainer)
    print("\n配置 Training Arguments...")
    use_early_stop = int(args.early_stopping_patience) > 0
    save_strategy = "epoch" if use_early_stop else "no"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=int(args.epochs),
        per_device_train_batch_size=int(args.train_batch_size),
        per_device_eval_batch_size=int(args.eval_batch_size),
        gradient_accumulation_steps=int(args.grad_accum),
        # 评估和保存策略
        evaluation_strategy="epoch",
        save_strategy=save_strategy,
        save_total_limit=1 if use_early_stop else None,
        logging_steps=50,
        # 优化器参数
        learning_rate=float(args.lr),
        weight_decay=float(args.weight_decay),
        warmup_ratio=float(args.warmup_ratio),
        label_smoothing_factor=float(args.label_smoothing),
        # 混合精度训练
        bf16=True if device == "cuda" else False,
        dataloader_num_workers=8,
        # 保存表现最好的模型权重 (关闭以节省空间)
        load_best_model_at_end=True if use_early_stop else False,
        metric_for_best_model="eval_loss" if use_early_stop else None,
        greater_is_better=False if use_early_stop else None,
        save_only_model=True,
        seed=int(args.seed),
    )

    # 初始化 Trainer
    callbacks = [EarlyStoppingCallback(early_stopping_patience=int(args.early_stopping_patience))] if use_early_stop else None
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=target_train,
        eval_dataset=target_test,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # 6. 执行训练循环
    print("\n🚀 开始训练 Target 模型 (前向传播 -> 计算 Loss -> 反向传播 -> 更新权重)...")
    trainer.train()

    # 7. 保存模型
    print(f"\n🎉 训练完成！正在将最优模型保存至: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()
