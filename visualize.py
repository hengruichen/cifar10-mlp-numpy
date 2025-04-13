import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


def plot_training_curves(history_path='logs/history.pkl', save_dir='figures', tag='default'):
    """
    绘制训练/验证损失 + 准确率曲线（包括 train_acc）
    """
    os.makedirs(save_dir, exist_ok=True)

    with open(history_path, 'rb') as f:
        history = pickle.load(f)

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss 曲线
    plt.figure()
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Loss ({tag})")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'loss_curve_{tag}.png'))
    plt.close()

    # Accuracy 曲线（新增 Train Accuracy 曲线）
    plt.figure()
    plt.plot(epochs, history['train_acc'], label='Train Accuracy', color='blue')
    plt.plot(epochs, history['val_acc'], label='Val Accuracy', color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Train vs Val Accuracy ({tag})")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'accuracy_curve_{tag}.png'))
    plt.close()

    print(f"训练曲线保存完成于 {save_dir}/loss_curve_{tag}.png 和 accuracy_curve_{tag}.png")



def visualize_weights(weight_path='weights/best_model.pkl',
                      save_dir='figures',
                      num_filters=16,
                      tag='default'):
    """
    可视化第一层权重 W1（输入维度为 3072，即 32x32x3）
    """
    os.makedirs(save_dir, exist_ok=True)

    with open(weight_path, 'rb') as f:
        params = pickle.load(f)

    W1 = params['W1'].T  # shape: (num_filters, 3072)

    fig, axes = plt.subplots(2, num_filters // 2, figsize=(15, 4))
    for i, ax in enumerate(axes.flat):
        if i >= W1.shape[0]:
            break
        w = W1[i].reshape(32, 32, 3)
        w = (w - np.min(w)) / (np.max(w) - np.min(w))  # normalize to [0, 1]
        ax.imshow(w)
        ax.axis('off')
        ax.set_title(f'Filter {i}')

    plt.suptitle(f"First Layer Weights Visualization ({tag})")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'weights_visualization_{tag}.png'))
    plt.close()

    print(f"第一层权重图保存完成于 {save_dir}/weights_visualization_{tag}.png")


# 示例用法（可在命令行快速调用）
if __name__ == "__main__":
    plot_training_curves(tag='he')
    visualize_weights(tag='he')
