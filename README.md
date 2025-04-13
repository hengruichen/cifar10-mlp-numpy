# 三层全连接神经网络（MLP）实现与实验报告

本项目旨在从零实现一个模块化的三层全连接神经网络（MLP），并在 CIFAR-10 图像分类任务上进行系统性评估与可视化分析。该实现**完全基于 NumPy**，不依赖任何深度学习框架，支持多种结构、激活函数、权重初始化方式与正则化设置，并结合训练过程中的损失、准确率和训练耗时进行全面对比分析。

---

## 项目结构

```
dl_homework/
├── figures/                     # 所有可视化输出图像
├── logs/                        # 保存搜索结果的CSV和日志
├── visualize_search/            # 高级可视化脚本（结构/热力图等）
├── weights/                     # 保存的模型权重（.pkl）这里只上传了默认配置权重和最优结果权重
├── config.yaml                  # 全局配置文件（结构/超参数等）
├── data_loader.py               # CIFAR-10数据加载与预处理
├── loss_fn.py                   # 交叉熵损失函数实现
├── model.py                     # 三层MLP模型结构（前向与反向传播）
├── prepare_npy.py               # 将CIFAR-10处理成.npy格式,但实际不需要
├── run_all.sh                   # Linux/Mac下一键运行脚本
├── search.py                    # 超参数组合搜索模块
├── search.log                   # 搜索日志输出
├── test.py                      # 加载模型进行评估
├── test_acc_plot.py             # 绘制准确率、耗时等分析图
├── train.py                     # 训练主程序
├── visualize.py                 # 训练权重/结构可视化脚本
└── README.md                    # 项目说明文档（即本文件）
```

---
## 数据集下载链接
https://www.kaggle.com/datasets/oxcdcd/cifar10
---

## 安装依赖

仅依赖 NumPy、Pandas、Matplotlib、Seaborn：

```bash
pip install numpy pandas matplotlib seaborn
```

---

## 快速开始

### 1. 直接运行run_all.sh

```bash
bash run_all.sh
```

### 2. 训练模型（默认配置在config.yaml中修改）

```bash
python train.py
```

默认配置：
- 网络结构：[256, 128, 64]
- 激活函数：ReLU
- 初始化方式：He
- 学习率：0.01，衰减率：0.95
- 正则化强度：0.0001
- Batch Size：64
- Epoch：70

### 3. 模型评估

```bash
python test.py
```

使用保存在 `weights/best_model.pkl` 的最优模型进行评估。

### 4. 超参数搜索（结构、激活函数、初始化等）

```bash
python search.py
```

所有实验结果将保存为 CSV：

- `logs/search_results.csv`

### 5. 可视化图表生成

```bash
python test_acc_plot.py
python visualize.py
```

图表将保存在 `figures/` 目录，包括准确率曲线、训练耗时、初始化热力图等。

---

## 最佳实验结果

在结构为 `[512, 256, 128]`，使用 ReLU 激活、He 初始化、不使用正则化的配置下，验证集准确率达到 **55.06%**，测试集准确率为 **54.63%**，是当前搜索空间下的最优方案。

---

## 模型与报告链接
- 模型权重下载：
  - [Google Drive](https://drive.google.com/file/d/1kd8AyFEj3NPQKLPyC8p98ZpkA8or2O1D/view?usp=drive_link)
---
