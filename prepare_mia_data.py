import argparse
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset
from transformers import AutoTokenizer


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v == "":
        return int(default)
    try:
        return int(v)
    except ValueError:
        return int(default)


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or v == "":
        return float(default)
    try:
        return float(v)
    except ValueError:
        return float(default)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--total-samples", type=int, default=_env_int("TOTAL_SAMPLES", 100000))
    p.add_argument("--target-total", type=int, default=_env_int("TARGET_TOTAL_SAMPLES", 10000))
    p.add_argument("--target-test-ratio", type=float, default=_env_float("TARGET_TEST_RATIO", 0.2))
    p.add_argument("--shadow-start", type=int, default=_env_int("SHADOW_START", 50000))
    p.add_argument("--shadow-total", type=int, default=_env_int("SHADOW_TOTAL_SAMPLES", 50000))
    p.add_argument("--shadow-test-ratio", type=float, default=_env_float("SHADOW_TEST_RATIO", 0.2))
    p.add_argument("--max-length", type=int, default=_env_int("MAX_LENGTH", 256))
    p.add_argument("--seed", type=int, default=_env_int("DATA_SEED", 42))
    return p

def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    print("=== 第一步：下载并处理 Concode 数据集 (完整实验版) ===")
    print("正在从 Hugging Face 下载 CodeXGLUE Text-to-Code (Concode) 数据集...")
    dataset = load_dataset("code_x_glue_tc_text_to_code")
    train_data = dataset["train"]
    print(f"成功加载训练集，共 {len(train_data)} 条样本。")

    print("\n=== 第二步：严格划分数据集（模拟真实黑盒攻击） ===")
    print(f"打乱数据集，并提取 {args.total_samples} 条子集用于真实实验...")
    shuffled_data = train_data.shuffle(seed=int(args.seed))
    
    target_total = int(args.target_total)
    shadow_start = int(args.shadow_start)
    shadow_total = int(args.shadow_total)

    target_data_full = shuffled_data.select(range(target_total))
    shadow_data_full = shuffled_data.select(range(shadow_start, shadow_start + shadow_total))

    target_splits = target_data_full.train_test_split(test_size=float(args.target_test_ratio), seed=int(args.seed))
    target_train = target_splits['train']
    target_test = target_splits['test']
    
    # Shadow 模型池划分: 由于我们要训练多个 Shadow 模型，这里提取一个总体池
    # 后续在 train_shadow_models.py 中，每个影子模型会从这个池子中随机采样（增加影子模型的多样性）
    # 但为了提取特征阶段有一致的“成员/非成员”基准，我们先在这里统一划分一个基础的 Shadow Train/Test
    shadow_splits = shadow_data_full.train_test_split(test_size=float(args.shadow_test_ratio), seed=int(args.seed))
    shadow_train = shadow_splits['train']
    shadow_test = shadow_splits['test']
    
    print(f"\n划分结果如下 (严格无交集)：")
    print(f"  [目标模型 Target] Train: {len(target_train)} 条, Test: {len(target_test)} 条")
    print(f"  [影子模型 Shadow] Train: {len(shadow_train)} 条, Test: {len(shadow_test)} 条 (作为基础池)")

    print("\n=== 第三步：数据预处理 (Tokenization) ===")
    print("加载 CodeGPT Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeGPT-small-java-adaptedGPT2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        inputs = [nl + "\n" + code for nl, code in zip(examples["nl"], examples["code"])]
        return tokenizer(inputs, padding="max_length", truncation=True, max_length=int(args.max_length))

    print("开始 Tokenize 目标模型数据...")
    target_train_tokenized = target_train.map(tokenize_function, batched=True, remove_columns=target_train.column_names)
    target_test_tokenized = target_test.map(tokenize_function, batched=True, remove_columns=target_test.column_names)
    
    print("开始 Tokenize 影子模型数据...")
    shadow_train_tokenized = shadow_train.map(tokenize_function, batched=True, remove_columns=shadow_train.column_names)
    shadow_test_tokenized = shadow_test.map(tokenize_function, batched=True, remove_columns=shadow_test.column_names)

    print("\n=== 第四步：保存处理好的数据集到本地 ===")
    output_dir = "mia_data"
    os.makedirs(output_dir, exist_ok=True)
    
    target_train_tokenized.save_to_disk(os.path.join(output_dir, "target_train"))
    target_test_tokenized.save_to_disk(os.path.join(output_dir, "target_test"))
    shadow_train_tokenized.save_to_disk(os.path.join(output_dir, "shadow_train"))
    shadow_test_tokenized.save_to_disk(os.path.join(output_dir, "shadow_test"))
    
    # 同时保存原始划分好的未 Tokenize 数据（为了第五步提取 AST 更加准确和方便，避免再按索引还原）
    raw_output_dir = "mia_data_raw"
    os.makedirs(raw_output_dir, exist_ok=True)
    target_train.save_to_disk(os.path.join(raw_output_dir, "target_train_raw"))
    target_test.save_to_disk(os.path.join(raw_output_dir, "target_test_raw"))
    shadow_train.save_to_disk(os.path.join(raw_output_dir, "shadow_train_raw"))
    shadow_test.save_to_disk(os.path.join(raw_output_dir, "shadow_test_raw"))

    print(f"成功！Tokenized 数据保存在 ./{output_dir}/，原始文本数据保存在 ./{raw_output_dir}/")

if __name__ == "__main__":
    main()
