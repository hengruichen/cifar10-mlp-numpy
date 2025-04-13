import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from model import MLP
from data_loader import load_cifar10
from loss_fn import compute_accuracy
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
# 读取 CIFAR-10 测试集
_, _, _, _, X_test, y_test = load_cifar10(**config['data'])

# 加载搜索结果 CSV
csv_path = "./logs/search_results.csv"
df = pd.read_csv(csv_path)

# 初始化结果列表
test_accuracies = []

# 遍历每个模型，评估 test acc
for idx, row in df.iterrows():
    weight_path = row['weight_path']
    try:
        with open(weight_path, 'rb') as f:
            params = pickle.load(f)

        # 构造模型
        model = MLP(
                input_dim=3072,
                hidden_dim1=int(row['h1']),
                hidden_dim2=int(row['h2']),
                hidden_dim3=int(row['h3']),
                output_dim=10,
                activation=row['activation'],
                init_method=row['init']
            )
        model.params = params

        logits, _ = model.forward(X_test)
        acc = compute_accuracy(logits, y_test)
    except Exception as e:
        print(f"[Error] Failed on {weight_path}: {e}")
        acc = 0.0
    test_accuracies.append(acc)

# 添加 test accuracy 列
df['test_acc'] = test_accuracies

# 直接覆盖写入原 CSV
df.to_csv(csv_path, index=False)
print(f"✅ Test accuracy added and saved to {csv_path}")

# 找出最高 / 最低点索引
min_idx = df['test_acc'].idxmin()
max_idx = df['test_acc'].idxmax()

# 提取对应信息
def get_label(row):
    base = os.path.basename(row['weight_path'])
    return f"{row['init']}_{row['activation']}_{int(row['h1'])}_{int(row['h2'])}_{int(row['h3'])}_lr{row['lr']}_reg{row['reg']}"

min_label = f"Min: {df.loc[min_idx, 'test_acc']:.2f}%\n" + get_label(df.loc[min_idx])
max_label = f"Max: {df.loc[max_idx, 'test_acc']:.2f}%\n" + get_label(df.loc[max_idx])

# 可视化
plt.figure(figsize=(13, 6))
plt.plot(df['test_acc'], marker='o', label='Test Accuracy')

# 标注线
plt.axhline(df['test_acc'][min_idx], color='red', linestyle='--', label='Min Accuracy')
plt.axhline(df['test_acc'][max_idx], color='green', linestyle='--', label='Max Accuracy')

# 添加注释
plt.annotate(min_label, (min_idx, df['test_acc'][min_idx]),
             textcoords="offset points", xytext=(0,10), ha='center', color='red', fontsize=9)
plt.annotate(max_label, (max_idx, df['test_acc'][max_idx]),
             textcoords="offset points", xytext=(0,10), ha='center', color='green', fontsize=9)

# 图信息
plt.title("Test Accuracy Across Different Hyperparameter Settings")
plt.xlabel("Experiment Index")
plt.ylabel("Test Accuracy (%)")
plt.grid(True)
plt.legend()
plt.tight_layout()

# 保存图片
plot_path = "./logs/test_accuracy_plot.png"
plt.savefig(plot_path)
print(f"📊 Accuracy plot saved to {plot_path}")
plt.show()
