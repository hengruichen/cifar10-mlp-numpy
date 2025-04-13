# search.py（支持断点续训）

import os
import csv
import time
import itertools
import yaml
import pickle

from model import MLP
from data_loader import load_cifar10
from train import train_model
from loss_fn import compute_loss, compute_accuracy

# 读取配置文件
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 更小的搜索空间（每层2个值）
space = config['search']
structure_list = [
    (128, 64, 32),
    (256, 128, 64),
    (512, 256, 128)
]
lr_list = config['search']['learning_rate']
reg_list = config['search']['weight_decay']
init_methods = config['search']['init_method']
activations = config['search']['activation']

search_space = list(itertools.product(
    structure_list, lr_list, reg_list, init_methods, activations
))

X_train, y_train, X_val, y_val, _, _ = load_cifar10(**config['data'])

os.makedirs("logs", exist_ok=True)
os.makedirs("weights", exist_ok=True)

csv_path = "logs/search_results.csv"
completed_tags = set()

# 如果存在旧结果，加载已完成组合（断点续训）
if os.path.exists(csv_path):
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        for row in reader:
            completed_tags.add(row[12].split("/")[-1].replace("best_model_", "").replace(".pkl", ""))
        print(completed_tags)

with open(csv_path, "a", newline='') as f:
    writer = csv.writer(f)
    if not completed_tags:
        writer.writerow([
            "h1", "h2", "h3", "lr", "reg", "init", "activation","train_acc", "train_loss",
            "val_acc", "val_loss", "time", "weight_path", "log_path"
        ])

    for (h1, h2, h3), lr, reg, init, act in search_space:
        tag = f"{init}_{act}_{h1}_{h2}_{h3}_lr{lr}_reg{reg}"
        if tag in completed_tags:
            print(f"跳过已完成组合: {tag}")
            continue

        print(f"\nSearching: {tag}")
        weight_path = f"weights/best_model_{tag}.pkl"
        log_path = f"logs/history_{tag}.pkl"

        model = MLP(
            input_dim=config['model']['input_dim'],
            hidden_dim1=h1,
            hidden_dim2=h2,
            hidden_dim3=h3,
            output_dim=config['model']['output_dim'],
            activation=act,
            init_method=init
        )

        start_time = time.time()
        val_acc, val_loss = train_model(
            model, X_train, y_train, X_val, y_val,
            lr=lr, reg=reg,
            num_epochs=config['train']['num_epochs'],
            batch_size=config['train']['batch_size'],
            lr_decay=config['train']['lr_decay'],
            save_path=weight_path,
            log_path=log_path
        )
        with open(log_path, 'rb') as log_f:
            history = pickle.load(log_f)
        train_loss = history['train_loss'][-1]
        train_acc = history['train_acc'][-1]
        elapsed = round(time.time() - start_time, 2)
        print(f"完成 {tag} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
                f"Time: {elapsed}s")

        writer.writerow([h1, h2, h3, lr, reg, init, act, train_acc, train_loss,val_acc, val_loss, elapsed, weight_path, log_path])
