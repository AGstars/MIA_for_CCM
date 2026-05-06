import os
import warnings
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import argparse
import torch
import pandas as pd
import numpy as np
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import tree_sitter_java as tsjava
from tree_sitter import Language, Parser
from typing import List, Dict, Tuple, Optional
import gc
import random

# ===================== 【核心：DEBUG 开关】 =====================
DEBUG_MODE = False          # 调试模式：True=仅用100条数据测试 | False=正式运行全量数据
DEBUG_SAMPLE_NUM = 100     # 调试模式采样数量

# ===================== 配置项 =====================
NUM_SHADOW_MODELS = int(os.environ.get("NUM_SHADOW_MODELS", "5"))

CONFIG = {
    "shadow_model_dirs": [f"models/shadow_model_{i}" for i in range(NUM_SHADOW_MODELS)],
    "target_model_dir": "models/target_model",
    "data_dir": "mia_data",
    "raw_data_dir": "mia_data_raw",
    "batch_size": 32,
    "max_seq_len": 128,
    "num_masks_per_sample": 5,
    "random_seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

def _split_csv(text: Optional[str]) -> List[str]:
    if not text:
        return []
    out: List[str] = []
    for p in str(text).split(","):
        p = p.strip()
        if p:
            out.append(p)
    return out

# ===================== 工具函数 =====================
def validate_input(text: Optional[str], default: str = "") -> str:
    if text is None:
        return default
    if not isinstance(text, str):
        try:
            text = str(text)
        except:
            return default
    text = text.strip().encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    return text if text else default

def get_default_model_features() -> Dict[str, float]:
    """单模型默认特征"""
    d = {
        "perplexity": 1e6,
        "loss_mean": 0.0, "loss_std": 0.0,
        "max_prob_mean": 0.0, "max_prob_std": 0.0,
        "entropy_mean": 0.0, "entropy_std": 0.0
    }
    for i in range(16):
        d[f"emb_{i}"] = 0.0
    return d

# ===================== AST特征提取 =====================
def extract_ast_features(code: str, parser: Parser) -> Dict[str, float]:
    try:
        code = validate_input(code)
        if not code:
            return {
                "ast_node_count":0,"ast_error_count":0,"ast_max_depth":0,"ast_cyclomatic_complexity":0,
                "ast_method_count":0,"ast_identifier_count":0,"ast_literal_count":0,"ast_loop_count":0,
                "ast_branch_count":0,"ast_exception_count":0,"ast_error_ratio":0.0,"ast_literal_ratio":0.0,"ast_identifier_ratio":0.0
            }
        source_bytes = code.encode("utf8", errors="ignore")
        tree = parser.parse(source_bytes)
        root_node = tree.root_node
        ast = {"node_count":0,"error_count":0,"max_depth":0,"cyclomatic":0,"method":0,"id":0,"lit":0,"loop":0,"branch":0,"exc":0}
        def trav(n, d):
            ast["node_count"]+=1
            ast["max_depth"]=max(ast["max_depth"],d)
            if n.is_error or n.is_missing: ast["error_count"]+=1
            t = n.type
            if t in ["if_statement","switch_statement","ternary_expression"]: ast["branch"]+=1; ast["cyclomatic"]+=1
            elif t in ["for_statement","while_statement","do_statement"]: ast["loop"]+=1; ast["cyclomatic"]+=1
            elif t == "catch_clause": ast["exc"]+=1; ast["cyclomatic"]+=1
            if t in ["method_declaration"]: ast["method"]+=1
            elif t == "identifier": ast["id"]+=1
            elif t in ["string_literal","integer_literal"]: ast["lit"]+=1
            for c in n.children: trav(c,d+1)
        trav(root_node,1)
        n = max(ast["node_count"],1)
        return {
            "ast_node_count":ast["node_count"],"ast_error_count":ast["error_count"],"ast_max_depth":ast["max_depth"],
            "ast_cyclomatic_complexity":ast["cyclomatic"],"ast_method_count":ast["method"],"ast_identifier_count":ast["id"],
            "ast_literal_count":ast["lit"],"ast_loop_count":ast["loop"],"ast_branch_count":ast["branch"],
            "ast_exception_count":ast["exc"],"ast_error_ratio":ast["error_count"]/n,"ast_literal_ratio":ast["lit"]/n,
            "ast_identifier_ratio":ast["id"]/n
        }
    except:
        return {k:0.0 for k in ["ast_node_count","ast_error_count","ast_max_depth","ast_cyclomatic_complexity","ast_method_count","ast_identifier_count","ast_literal_count","ast_loop_count","ast_branch_count","ast_exception_count","ast_error_ratio","ast_literal_ratio","ast_identifier_ratio"]}

# ===================== 单模型特征提取（修复：逐样本独立计算，无重复） =====================
def extract_single_model_features(model, tokenizer, text: str, device):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=CONFIG["max_seq_len"], padding=True).to(device)
        input_ids = inputs["input_ids"]
        
        with torch.no_grad():
            outputs = model(**inputs, labels=input_ids, output_hidden_states=True)
            loss = outputs.loss.item()
            logits = outputs.logits
            hidden_states = outputs.hidden_states[-1] # 取最后一层隐藏状态
            
        ppl = torch.exp(torch.tensor(loss)).item()
        
        # 计算序列平均池化的 Embedding 向量
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        valid_hidden = hidden_states * attention_mask
        sum_hidden = valid_hidden.sum(dim=1)
        sum_mask = attention_mask.sum(dim=1).clamp(min=1e-9)
        seq_embedding = (sum_hidden / sum_mask).squeeze(0).cpu().numpy()
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        valid = shift_labels != tokenizer.pad_token_id
        if valid.sum() == 0:
            return get_default_model_features()
            
        valid_logits = shift_logits[valid]
        probs = torch.softmax(valid_logits, dim=-1)
        
        max_probs = probs.max(dim=-1).values
        entropies = (-probs * torch.log(probs + 1e-10)).sum(dim=-1)
        
        feat_dict = {
            "perplexity": ppl,
            "loss_mean": loss,
            "loss_std": 0.0,
            "max_prob_mean": max_probs.mean().item(),
            "max_prob_std": max_probs.std().item(),
            "entropy_mean": entropies.mean().item(),
            "entropy_std": entropies.std().item()
        }
        
        # 保存前 16 个维度的 embedding 作为高级特征 (全部保存会导致维度过高)
        for i in range(16):
            feat_dict[f"emb_{i}"] = seq_embedding[i].item()
            
        return feat_dict
    except: pass
    return get_default_model_features()

# ===================== 似然比 (LiRA) 特征提取 =====================
def extract_lira_features(eval_model_tuple, ref_models_tuples, nl: str, code: str, device):
    text = validate_input(nl) + "\n" + validate_input(code)
    
    # 目标/被评估模型的特征
    eval_m, eval_t = eval_model_tuple
    eval_feats = extract_single_model_features(eval_m, eval_t, text, device)
    
    # 参考影子模型的特征
    ref_feats_list = []
    for m, t in ref_models_tuples:
        ref_feats_list.append(extract_single_model_features(m, t, text, device))
        
    fused = {}
    eps = 1e-8
    for k in ["perplexity", "loss_mean", "max_prob_mean", "entropy_mean"] + [f"emb_{i}" for i in range(16)]:
        fused[f"eval_{k}"] = round(eval_feats[k], 6)

        if not ref_feats_list:
            ref_mean = float(eval_feats[k])
            ref_std = 0.0
        else:
            ref_vals = [f[k] for f in ref_feats_list]
            ref_mean = float(np.mean(ref_vals))
            ref_std = float(np.std(ref_vals))
        fused[f"ref_{k}_mean"] = round(ref_mean, 6)
        fused[f"ref_{k}_std"] = round(ref_std, 6)
        
        # LiRA 核心：差异特征 (例如 eval_loss - ref_loss_mean)
        diff = float(eval_feats[k] - ref_mean)
        fused[f"lira_diff_{k}"] = round(diff, 6)

        fused[f"lira_z_{k}"] = round(diff / (ref_std + eps), 6)
        
    return fused

# ===================== 预加载模型 =====================
def preload_models():
    ms = []
    print(f"🔹 预加载 {len(CONFIG['shadow_model_dirs'])} 个影子模型...")
    for i, d in enumerate(CONFIG["shadow_model_dirs"]):
        print(f"   加载模型 {i+1}/{len(CONFIG['shadow_model_dirs'])}：{d}")
        t = AutoTokenizer.from_pretrained(d)
        if t.pad_token is None: t.pad_token = t.eos_token
        m = AutoModelForCausalLM.from_pretrained(d).to(CONFIG["device"]).eval()
        ms.append((m, t))
        
    print("🔹 预加载目标模型...")
    t_target = AutoTokenizer.from_pretrained(CONFIG["target_model_dir"])
    if t_target.pad_token is None: t_target.pad_token = t_target.eos_token
    m_target = AutoModelForCausalLM.from_pretrained(CONFIG["target_model_dir"]).to(CONFIG["device"]).eval()
    
    print("✅ 所有模型加载完成！\n")
    return ms, (m_target, t_target)

# ===================== 数据集处理（支持DEBUG截断 + 1:1均衡） =====================
def process_balanced_dataset(positive_ds, negative_ds, raw_pos, raw_neg, label_pos, label_neg, parser, eval_model, ref_models):
    # DEBUG 模式：仅取少量数据测试
    if DEBUG_MODE:
        max_sample = DEBUG_SAMPLE_NUM // 2
        positive_ds = positive_ds.select(range(min(len(positive_ds), max_sample)))
        negative_ds = negative_ds.select(range(min(len(negative_ds), max_sample)))
        raw_pos = raw_pos.select(range(min(len(raw_pos), max_sample)))
        raw_neg = raw_neg.select(range(min(len(raw_neg), max_sample)))

    min_len = min(len(positive_ds), len(negative_ds))
    print(f"📊 样本均衡：正样本{min_len} | 负样本{min_len} | 总计{min_len*2}条")

    pos_feats, neg_feats = [], []
    # 处理正样本
    for i in tqdm(range(min_len), desc="正样本(成员)"):
        nl, code = raw_pos[i]["nl"], raw_pos[i]["code"]
        ast = extract_ast_features(code, parser)
        model_feats = extract_lira_features(eval_model, ref_models, nl, code, CONFIG["device"])
        pos_feats.append({**model_feats, **ast, "is_member": label_pos})
        # 清理显存
        if CONFIG["device"] == "cuda": torch.cuda.empty_cache()

    # 处理负样本
    for i in tqdm(range(min_len), desc="负样本(非成员)"):
        nl, code = raw_neg[i]["nl"], raw_neg[i]["code"]
        ast = extract_ast_features(code, parser)
        model_feats = extract_lira_features(eval_model, ref_models, nl, code, CONFIG["device"])
        neg_feats.append({**model_feats, **ast, "is_member": label_neg})
        # 清理显存
        if CONFIG["device"] == "cuda": torch.cuda.empty_cache()

    all_feats = pos_feats + neg_feats
    random.shuffle(all_feats)
    return all_feats

# ===================== 主函数 =====================
def main(argv: Optional[List[str]] = None):
    global DEBUG_MODE
    global DEBUG_SAMPLE_NUM

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--data-dir", default=CONFIG["data_dir"])
    parser.add_argument("--raw-data-dir", default=CONFIG["raw_data_dir"])
    parser.add_argument("--target-model-dir", default=CONFIG["target_model_dir"])
    parser.add_argument("--shadow-model-dirs", default="")
    parser.add_argument("--shadow-model-root", default="")
    parser.add_argument("--num-shadow-models", type=int, default=NUM_SHADOW_MODELS)
    parser.add_argument("--batch-size", type=int, default=int(CONFIG["batch_size"]))
    parser.add_argument("--max-seq-len", type=int, default=int(CONFIG["max_seq_len"]))
    parser.add_argument("--num-masks-per-sample", type=int, default=int(CONFIG["num_masks_per_sample"]))
    parser.add_argument("--seed", type=int, default=int(CONFIG["random_seed"]))
    parser.add_argument("--device", default=CONFIG["device"])
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-sample-num", type=int, default=DEBUG_SAMPLE_NUM)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--output-train-csv", default="")
    parser.add_argument("--output-test-csv", default="")
    parser.add_argument("--suffix", default="")
    args = parser.parse_args(argv)

    CONFIG["data_dir"] = args.data_dir
    CONFIG["raw_data_dir"] = args.raw_data_dir
    CONFIG["target_model_dir"] = args.target_model_dir
    CONFIG["batch_size"] = int(args.batch_size)
    CONFIG["max_seq_len"] = int(args.max_seq_len)
    CONFIG["num_masks_per_sample"] = int(args.num_masks_per_sample)
    CONFIG["random_seed"] = int(args.seed)
    CONFIG["device"] = args.device

    shadow_dirs = _split_csv(args.shadow_model_dirs)
    if shadow_dirs:
        CONFIG["shadow_model_dirs"] = shadow_dirs
    elif args.shadow_model_root:
        root = args.shadow_model_root
        CONFIG["shadow_model_dirs"] = [os.path.join(root, f"shadow_model_{i}") for i in range(int(args.num_shadow_models))]
    else:
        CONFIG["shadow_model_dirs"] = [f"models/shadow_model_{i}" for i in range(int(args.num_shadow_models))]

    DEBUG_MODE = bool(args.debug)
    DEBUG_SAMPLE_NUM = int(args.debug_sample_num)

    random.seed(CONFIG["random_seed"])
    np.random.seed(CONFIG["random_seed"])
    torch.manual_seed(CONFIG["random_seed"])

    print("="*60)
    print(f"🚀 MIA 攻击特征提取 | DEBUG模式：{DEBUG_MODE} | 设备：{CONFIG['device']}")
    print("="*60)

    # 1. 初始化AST解析器
    parser = Parser()
    try:
        lang = Language(tsjava.language(), "java")
    except TypeError:
        try:
            lang = Language(tsjava.language())
        except Exception:
            lang = tsjava.language()
    if hasattr(parser, "set_language"):
        parser.set_language(lang)
    else:
        parser.language = lang

    # 2. 加载数据集
    print("🔹 加载数据集...")
    shadow_train = load_from_disk(f"{CONFIG['data_dir']}/shadow_train")
    shadow_test = load_from_disk(f"{CONFIG['data_dir']}/shadow_test")
    target_train = load_from_disk(f"{CONFIG['data_dir']}/target_train")
    target_test = load_from_disk(f"{CONFIG['data_dir']}/target_test")

    raw_strain = load_from_disk(f"{CONFIG['raw_data_dir']}/shadow_train_raw")
    raw_stest = load_from_disk(f"{CONFIG['raw_data_dir']}/shadow_test_raw")
    raw_ttrain = load_from_disk(f"{CONFIG['raw_data_dir']}/target_train_raw")
    raw_ttest = load_from_disk(f"{CONFIG['raw_data_dir']}/target_test_raw")

    # 3. 预加载所有模型
    shadows, target = preload_models()

    # --------------------------
    # 生成攻击训练集
    # --------------------------
    print("\n🔹 生成攻击训练集...")
    # 对于影子数据集，正样本(成员)使用影子模型的训练集，负样本(非成员)使用影子模型的测试集
    # 这里的数量由 min(len(shadow_train), len(shadow_test)) 决定
    attack_train = process_balanced_dataset(shadow_train, shadow_test, raw_strain, raw_stest, 1, 0, parser, shadows[0], shadows[1:])
    
    # --------------------------
    # 生成攻击测试集
    # --------------------------
    print("\n🔹 生成攻击测试集...")
    # 对于目标数据集，正样本(成员)使用目标模型的训练集，负样本(非成员)使用目标模型的测试集
    # 这里的数量由 min(len(target_train), len(target_test)) 决定
    attack_test = process_balanced_dataset(target_train, target_test, raw_ttrain, raw_ttest, 1, 0, parser, target, shadows)

    os.makedirs(args.output_dir, exist_ok=True)

    # 4. 保存文件（DEBUG/正式 自动区分）
    print("\n💾 保存数据集...")
    suffix = args.suffix
    if not suffix:
        suffix = "_debug" if DEBUG_MODE else ""
    
    df_train = pd.DataFrame(attack_train)
    df_test = pd.DataFrame(attack_test)
    
    train_path = args.output_train_csv or os.path.join(args.output_dir, f"attack_train_dataset{suffix}.csv")
    test_path = args.output_test_csv or os.path.join(args.output_dir, f"attack_test_dataset{suffix}.csv")
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)

    # 5. 最终总结
    print("\n" + "="*60)
    print(f"🎉 处理完成！DEBUG模式：{DEBUG_MODE}")
    print(f"📁 训练集：{train_path} ({len(df_train)} 条)")
    print(f"📁 测试集：{test_path} ({len(df_test)} 条)")
    print(f"✅ 特征列：5模型融合(loss_mean_mean/perplexity_std等) + AST特征 + 标签")
    print(f"✅ 样本均衡：1:1 | 无特征重复 | 全模型使用")
    print("="*60)

    # 释放显存
    del shadows, target
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()
