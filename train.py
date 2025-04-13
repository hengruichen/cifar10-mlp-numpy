# train.py

import numpy as np
import os
import pickle
import yaml
from model import MLP
from data_loader import load_cifar10
from loss_fn import compute_loss, compute_accuracy

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def train_model(model, X_train, y_train, X_val, y_val,
                lr, reg, num_epochs, batch_size,
                lr_decay, save_path,log_path):

    num_train = X_train.shape[0]
    best_val_acc = 0
    best_model = None
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [],'val_acc': []}
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(num_epochs):
        indices = np.arange(num_train)
        np.random.shuffle(indices)
        X_train, y_train = X_train[indices], y_train[indices]

        for i in range(0, num_train, batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            logits, cache = model.forward(X_batch)
            loss, grads = model.backward(logits, y_batch, cache, reg)
            model.update_params(grads, lr)

        logits_train, _ = model.forward(X_train)
        train_loss, _ = model.backward(logits_train, y_train, model.forward(X_train)[1], reg)
        train_acc = compute_accuracy(logits_train, y_train)

        logits_val, _ = model.forward(X_val)
        val_loss = compute_loss(logits_val, y_val, reg, model.params)
        val_acc = compute_accuracy(logits_val, y_val)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"[Epoch {epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.params.copy()
            with open(save_path, 'wb') as f:
                pickle.dump(best_model, f)

        lr *= lr_decay

    # 保存训练日志
    with open(log_path, 'wb') as f:
        pickle.dump(history, f)

    return best_val_acc, val_loss


if __name__ == '__main__':
    X_train, y_train, X_val, y_val, _, _ = load_cifar10(**config['data'])
    model = MLP(**config['model'])
    train_model(
        model,
        X_train, y_train, X_val, y_val,
        lr=config['train']['learning_rate'],
        reg=config['train']['weight_decay'],
        num_epochs=config['train']['num_epochs'],
        batch_size=config['train']['batch_size'],
        lr_decay=config['train']['lr_decay'],
        save_path=config['train']['save_path'],
        log_path=config['train']['log_path']
    )
