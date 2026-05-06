import pandas as pd

# 读取生成的特征文件
df = pd.read_csv("attack_train_dataset_debug.csv")

print("=== 数据质量核查报告 ===")
print(f"总样本数: {len(df)}")

# 1. 检查值是否重复（最关键的一点）
# 我们检查 loss_mean_mean 这一列，看它有多少个唯一值
unique_losses = df['loss_mean_mean'].nunique()
print(f"loss_mean_mean 唯一值数量: {unique_losses} / {len(df)}")

if unique_losses <= 1:
    print("❌ 警告：该列所有值完全重复！说明 batch 计算逻辑仍存在平均化问题。")
elif unique_losses < len(df) * 0.9:
    print("⚠️ 提示：存在部分重复值，请检查是否存在重复代码样本。")
else:
    print("✅ 通过：每一行数据都有独特的 Loss 特征。")

# 2. 检查 Std 列
# 如果 std 为 0，说明 5 个影子模型跑出来的结果一模一样，融合失去了意义
avg_std = df['loss_mean_std'].mean()
print(f"影子模型间 Loss 的平均标准差: {avg_std:.6f}")

if avg_std == 0:
    print("❌ 警告：模型间的标准差为 0！请检查是否重复加载了同一个模型。")
else:
    print("✅ 通过：不同影子模型对同一数据的反应存在差异（鲁棒性特征有效）。")

# 3. 打印前 5 行观察
print("\n[预览前 5 行特征]:")
print(df[['loss_mean_mean', 'loss_mean_std', 'perplexity_mean', 'is_member']].head())