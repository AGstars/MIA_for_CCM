import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("提示: 未安装 xgboost 库，将只使用 RandomForest。若想使用 XGBoost，请运行 pip install xgboost")

def evaluate_model(name, model, X_test, y_test):
    """
    评估模型，计算并打印各项指标 (Accuracy, Precision, Recall, F1, AUC)
    以及高安全标准下的 TPR @ Low FPR
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    return evaluate_probabilities(name, y_prob, y_test)

def _scores_to_prob(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    mn = np.nanmin(scores)
    mx = np.nanmax(scores)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn < 1e-12:
        return np.full(scores.shape, 0.5, dtype=np.float64)
    probs = (scores - mn) / (mx - mn)
    probs = np.clip(probs, 0.0, 1.0)
    probs[np.isnan(probs)] = 0.5
    return probs

def evaluate_probabilities(name, y_prob, y_test):
    y_prob = np.asarray(y_prob, dtype=np.float64)
    
    # 1. 计算最佳 F1-Score 对应的阈值 (Threshold Tuning)
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    # 避免除以 0
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    
    # 使用优化后的阈值进行预测
    y_pred_opt = (y_prob >= best_threshold).astype(int)
    
    acc = accuracy_score(y_test, y_pred_opt)
    prec = precision_score(y_test, y_pred_opt, zero_division=0)
    rec = recall_score(y_test, y_pred_opt, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
    
    def get_tpr_at_fpr(target_fpr):
        eligible = np.where(fpr <= target_fpr)[0]
        if len(eligible) == 0:
            return float(tpr[0])
        return float(tpr[eligible[-1]])
        
    tpr_at_1_fpr = get_tpr_at_fpr(0.01)
    tpr_at_01_fpr = get_tpr_at_fpr(0.001)
    
    print(f"\n--- 【{name}】评估结果 ---")
    print(f"最佳分类阈值 (Best Threshold): {best_threshold:.4f}")
    print(f"Accuracy (准确率 @ {best_threshold:.2f}): {acc:.4f}")
    print(f"Precision (精确率 @ {best_threshold:.2f}): {prec:.4f}")
    print(f"Recall (召回率 @ {best_threshold:.2f}):    {rec:.4f}")
    print(f"F1-Score (最高 F1):  {best_f1:.4f}")
    print(f"ROC AUC (AUC值):    {auc:.4f}")
    print(f"TPR @ 1% FPR:       {tpr_at_1_fpr:.4f}  <-- 重要指标")
    print(f"TPR @ 0.1% FPR:     {tpr_at_01_fpr:.4f}  <-- 严苛指标")
    
    return {
        "Accuracy": acc,
        "F1": float(best_f1),
        "AUC": float(auc),
        "y_prob": y_prob,
        "fpr": fpr,
        "tpr": tpr,
        "precisions": precisions,
        "recalls": recalls,
        "thresholds_pr": thresholds,
        "f1_scores_pr": f1_scores,
        "best_threshold": float(best_threshold),
        "TPR@1%FPR": float(tpr_at_1_fpr),
        "TPR@0.1%FPR": float(tpr_at_01_fpr),
    }

def plot_metric_summary(results_dict, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    models = list(results_dict.keys())
    metrics = {
        "AUC": [results_dict[m]["AUC"] for m in models],
        "F1": [results_dict[m]["F1"] for m in models],
        "TPR@1%FPR": [results_dict[m]["TPR@1%FPR"] for m in models],
        "TPR@0.1%FPR": [results_dict[m]["TPR@0.1%FPR"] for m in models],
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for ax, (metric_name, vals) in zip(axes, metrics.items()):
        x = np.arange(len(models))
        bars = ax.bar(x, vals, color="slateblue", alpha=0.85)
        ax.set_title(metric_name)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.set_ylim(0.0, max(0.05, float(np.max(vals)) * 1.15))
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h, f"{h:.4f}", ha="center", va="bottom", fontsize=8)
        ax.grid(True, axis="y", alpha=0.2)

    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "metric_summary.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_threshold_diagnostics(results_dict, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    for name, res in results_dict.items():
        thresholds = res.get("thresholds_pr")
        f1_scores = res.get("f1_scores_pr")
        precisions = res.get("precisions")
        recalls = res.get("recalls")
        best_thr = res.get("best_threshold", 0.5)
        if thresholds is None or f1_scores is None or precisions is None or recalls is None:
            continue

        thr = np.asarray(thresholds, dtype=np.float64)
        f1 = np.asarray(f1_scores[: len(thr)], dtype=np.float64)
        p = np.asarray(precisions[: len(thr)], dtype=np.float64)
        r = np.asarray(recalls[: len(thr)], dtype=np.float64)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(thr, p, label="Precision", color="steelblue")
        ax.plot(thr, r, label="Recall", color="darkorange")
        ax.plot(thr, f1, label="F1", color="seagreen")
        ax.axvline(best_thr, color="red", linestyle="--", linewidth=1.2, label=f"Best thr={best_thr:.3f}")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_title(f"Threshold Diagnostics - {name}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        safe = "".join([c if c.isalnum() or c in "-_" else "_" for c in name])
        plt.savefig(os.path.join(output_dir, f"threshold_curve_{safe}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

def plot_confusion_matrices(results_dict, y_test, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    y_test = np.asarray(y_test, dtype=int)
    for name, res in results_dict.items():
        y_prob = np.asarray(res.get("y_prob"), dtype=np.float64)
        if y_prob is None or len(y_prob) != len(y_test):
            continue
        for thr_name, thr in [("default_0.5", 0.5), ("tuned", float(res.get("best_threshold", 0.5)))]:
            y_pred = (y_prob >= thr).astype(int)
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            fig, ax = plt.subplots(figsize=(4.5, 4.0))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_xticklabels(["Non-member", "Member"], rotation=0)
            ax.set_yticklabels(["Non-member", "Member"], rotation=0)
            ax.set_title(f"Confusion Matrix - {name}\n({thr_name}, thr={thr:.3f})")
            fig.tight_layout()
            safe = "".join([c if c.isalnum() or c in "-_" else "_" for c in name])
            plt.savefig(os.path.join(output_dir, f"confusion_{safe}_{thr_name}.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)

def plot_det_curve(results_dict, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    for name, res in results_dict.items():
        fpr = np.asarray(res.get("fpr"), dtype=np.float64)
        tpr = np.asarray(res.get("tpr"), dtype=np.float64)
        if fpr is None or tpr is None:
            continue
        fnr = 1.0 - tpr
        plt.plot(fpr, fnr, label=name)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim([1e-4, 1.0])
    plt.ylim([1e-3, 1.0])
    plt.xlabel("False Positive Rate (log)")
    plt.ylabel("False Negative Rate (log)")
    plt.title("DET Curve (log-log)")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(output_dir, "det_curve_loglog.png"), dpi=300, bbox_inches="tight")
    plt.close()

def plot_feature_importance(feature_cols, importances, title, output_path, top_k=20):
    feature_cols = list(feature_cols)
    importances = np.asarray(importances, dtype=np.float64)
    if len(importances) != len(feature_cols):
        return
    idx = np.argsort(importances)[::-1][:top_k]
    top_feats = [feature_cols[i] for i in idx]
    top_imps = importances[idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(top_feats))[::-1], top_imps[::-1], color="teal", alpha=0.85)
    ax.set_yticks(range(len(top_feats))[::-1])
    ax.set_yticklabels(top_feats[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_curves(results_dict, output_dir="plots"):
    """绘制 Log-scale ROC 和 PR 曲线"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 绘制 Log-scale ROC 曲线
    plt.figure(figsize=(8, 6))
    for name, res in results_dict.items():
        plt.plot(res["fpr"], res["tpr"], label=f'{name} (AUC = {res["AUC"]:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xscale('log')
    # 限制 X 轴的显示范围，避免从 0 开始报错
    plt.xlim([1e-4, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Log Scale)')
    plt.ylabel('True Positive Rate')
    plt.title('MIA ROC Curve (Log Scale)')
    plt.legend(loc="lower right")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(output_dir, 'roc_curve_log.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 绘制常规 ROC 曲线
    plt.figure(figsize=(8, 6))
    for name, res in results_dict.items():
        plt.plot(res["fpr"], res["tpr"], label=f'{name} (AUC = {res["AUC"]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('MIA ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 绘制 PR 曲线
    plt.figure(figsize=(8, 6))
    for name, res in results_dict.items():
        plt.plot(res["recalls"], res["precisions"], label=f'{name} (Best F1 = {res["F1"]:.4f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('MIA Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'pr_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 绘制 TPR @ Low FPR 的柱状图
    plt.figure(figsize=(10, 6))
    
    models = list(results_dict.keys())
    tpr_1_vals = [results_dict[m]["TPR@1%FPR"] for m in models]
    tpr_01_vals = [results_dict[m]["TPR@0.1%FPR"] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, tpr_1_vals, width, label='TPR @ 1% FPR', color='steelblue')
    rects2 = ax.bar(x + width/2, tpr_01_vals, width, label='TPR @ 0.1% FPR', color='darkorange')
    
    ax.set_ylabel('True Positive Rate')
    ax.set_title('TPR at Low FPR by Model/Feature Set')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    
    # 在柱子上添加数值标签
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
                        
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tpr_bar_chart.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    plt.close('all') # 确保之前的空 figure 也被关闭
    
    print(f"\n📈 已生成 Log-scale ROC 曲线、PR 曲线和 TPR柱状图，保存在 {output_dir}/ 目录下。")

def run_ablation_study(df_train, df_test, base_xgb_params):
    """
    执行特征消融实验 (Ablation Studies)，并保存结果和图表
    """
    print("\n" + "="*50)
    print("🔬 开始特征消融实验 (Ablation Studies)")
    print("="*50)
    
    # 定义不同的特征子集
    all_cols = [col for col in df_train.columns if col != 'is_member']
    
    # 1. 基础特征 (仅目标模型自身的 Loss/Perplexity，无影子模型参考)
    basic_features = ['eval_perplexity', 'eval_loss_mean', 'eval_max_prob_mean', 'eval_entropy_mean']
    
    # 2. 基础特征 + AST 句法特征
    ast_features = [col for col in all_cols if col.startswith('ast_')]
    basic_ast_features = basic_features + ast_features
    
    # 3. 基础特征 + Embedding 语义特征
    emb_features = [col for col in all_cols if col.startswith('eval_emb_')]
    basic_emb_features = basic_features + emb_features
    
    # 4. 全量特征 (加入 LiRA diff 和 ref 统计量)
    # 即 all_cols 减去被人工剔除的无效 AST
    ignored_ast = ['ast_error_count', 'ast_error_ratio', 'ast_method_count', 'ast_loop_count', 'ast_exception_count']
    full_features = [col for col in all_cols if col not in ignored_ast]
    
    feature_sets = {
        "1. Basic (Target Only)": basic_features,
        "2. Basic + AST": basic_ast_features,
        "3. Basic + Embedding": basic_emb_features,
        "4. Full (LiRA + Emb + AST)": full_features
    }
    
    ablation_results = {}
    
    y_train = df_train['is_member'].values
    y_test = df_test['is_member'].values
    
    for name, f_cols in feature_sets.items():
        print(f"\n⏳ 正在评估特征组: {name} (维度: {len(f_cols)})")
        X_train_sub = df_train[f_cols].values
        X_test_sub = df_test[f_cols].values
        
        clf = xgb.XGBClassifier(**base_xgb_params)
        clf.fit(X_train_sub, y_train)
        
        res = evaluate_model(f"Ablation: {name}", clf, X_test_sub, y_test)
        ablation_results[name] = res
        
    # 绘制消融实验的 ROC/PR 对比图
    plot_curves(ablation_results, output_dir="plots/ablation")
    plot_metric_summary(ablation_results, output_dir="plots/ablation")
    plot_threshold_diagnostics(ablation_results, output_dir="plots/ablation")
    plot_confusion_matrices(ablation_results, y_test, output_dir="plots/ablation")
    plot_det_curve(ablation_results, output_dir="plots/ablation")
    print("\n✅ 特征消融实验完成！图表已保存至 plots/ablation/ 目录。")

def main():
    print("=== 第六步：实施攻击验证 (MIA) ===")
    
    # 1. 加载数据集
    train_csv = "attack_train_dataset.csv"
    test_csv = "attack_test_dataset.csv"
    
    try:
        df_train = pd.read_csv(train_csv)
        df_test = pd.read_csv(test_csv)
        print(f"成功加载训练数据 (Shadow): {df_train.shape[0]} 条记录")
        print(f"成功加载测试数据 (Target): {df_test.shape[0]} 条记录")
    except FileNotFoundError:
        print("找不到攻击数据集 CSV，请先运行 extract_features.py 提取特征！")
        return

    # 2. 划分特征 (X) 和标签 (y)
    # 所有列除了 'is_member' 都是特征，且为了优化模型，移除所有重要性为 0.0 的弱 AST 特征
    ignored_ast_features = [
        'ast_error_count', 'ast_error_ratio', 
        'ast_method_count', 'ast_loop_count', 'ast_exception_count'
    ]
    feature_cols = [col for col in df_train.columns if col != 'is_member' and col not in ignored_ast_features]
    print(f"\n使用的特征维度 (共 {len(feature_cols)} 维): {feature_cols}")
    
    X_train = df_train[feature_cols].values
    y_train = df_train['is_member'].values
    
    X_test = df_test[feature_cols].values
    y_test = df_test['is_member'].values
    
    # 3. 定义训练并收集结果的逻辑 (为消融实验做准备)
    results = {}
    
    print("\n🚀 正在训练 Random Forest 分类器...")
    # 增加 n_estimators 和 max_depth 以适应更大的真实数据集
    rf_clf = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, n_jobs=16)
    rf_clf.fit(X_train, y_train)
    results["Random Forest"] = evaluate_model("Random Forest", rf_clf, X_test, y_test)

    if "eval_loss_mean" in df_test.columns:
        loss_score = -df_test["eval_loss_mean"].values
        results["Threshold: Target Loss"] = evaluate_probabilities(
            "Threshold: Target Loss",
            _scores_to_prob(loss_score),
            y_test,
        )

    if "lira_z_loss_mean" in df_test.columns:
        lira_z_score = -df_test["lira_z_loss_mean"].values
        results["Threshold: LiRA Z(Loss)"] = evaluate_probabilities(
            "Threshold: LiRA Z(Loss)",
            _scores_to_prob(lira_z_score),
            y_test,
        )
    
    # 打印特征重要性
    importances = rf_clf.feature_importances_
    print("\n[Random Forest 特征重要性]:")
    for name, imp in zip(feature_cols, importances):
        print(f"  - {name}: {imp:.4f}")

    plot_feature_importance(
        feature_cols,
        importances,
        title="Random Forest Feature Importance (Top 20)",
        output_path=os.path.join("plots", "main_models", "feature_importance_rf_top20.png"),
        top_k=20,
    )

    # 4. 训练并评估 XGBoost 分类器 (如果已安装)
    xgb_params = {
        'n_estimators': 300, 
        'max_depth': 8, 
        'learning_rate': 0.05, 
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42, 
        'eval_metric': 'logloss',
        'n_jobs': 16
    }
    
    if HAS_XGB:
        print("\n🚀 正在训练 XGBoost 分类器 (Full Features)...")
        xgb_clf = xgb.XGBClassifier(**xgb_params)
        xgb_clf.fit(X_train, y_train)
        results["XGBoost (Full)"] = evaluate_model("XGBoost (Full)", xgb_clf, X_test, y_test)
        
        # 打印特征重要性
        xgb_importances = xgb_clf.feature_importances_
        print("\n[XGBoost 特征重要性]:")
        for name, imp in zip(feature_cols, xgb_importances):
            print(f"  - {name}: {imp:.4f}")

        plot_feature_importance(
            feature_cols,
            xgb_importances,
            title="XGBoost Feature Importance (Top 20)",
            output_path=os.path.join("plots", "main_models", "feature_importance_xgb_top20.png"),
            top_k=20,
        )

    # 5. 绘制评估曲线 (全量特征对比)
    plot_curves(results, output_dir="plots/main_models")

    plot_metric_summary(results, output_dir="plots/main_models")
    plot_threshold_diagnostics(results, output_dir="plots/main_models")
    plot_confusion_matrices(results, y_test, output_dir="plots/main_models")
    plot_det_curve(results, output_dir="plots/main_models")

    # 6. 运行特征消融实验 (Ablation Studies)
    if HAS_XGB:
        run_ablation_study(df_train, df_test, xgb_params)

    print("\n🎉 攻击验证完成！")
    print("结论：如果 AUC 和 Accuracy 明显高于 0.5 (例如达到 0.65 甚至 0.8 以上)，说明成员推理攻击成功！目标模型泄漏了其训练数据的隐私。")

if __name__ == "__main__":
    main()
