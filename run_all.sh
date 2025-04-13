#!/bin/bash

# 创建所需目录（如果还没创建）
mkdir -p weights logs figures

# 训练模型
python train.py

# 测试模型
python test.py

# 可视化训练曲线和第一层权重图像
python -c "from visualize import plot_training_curves, visualize_weights; plot_training_curves(); visualize_weights()"

# 超参数搜索（交互式）
read -p "是否运行超参数搜索？(y/n): " choice
if [[ "$choice" == "y" ]]; then
  python search.py
fi

echo "所有任务执行完毕！请查看 logs/, weights/, figures/ 文件夹中的结果。"
