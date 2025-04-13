# prepare_npy.py

import numpy as np
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import os

def save_numpy_cifar10(save_dir='data'):
    os.makedirs(save_dir, exist_ok=True)

    transform = transforms.ToTensor()
    trainset = CIFAR10(root='./cifar10_raw', train=True, download=True, transform=transform)
    testset = CIFAR10(root='./cifar10_raw', train=False, download=True, transform=transform)

    X_train = np.stack([np.array(img[0].permute(1, 2, 0)) * 255 for img in trainset]).astype(np.uint8)
    y_train = np.array([img[1] for img in trainset])

    X_test = np.stack([np.array(img[0].permute(1, 2, 0)) * 255 for img in testset]).astype(np.uint8)
    y_test = np.array([img[1] for img in testset])

    np.save(os.path.join(save_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(save_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(save_dir, 'y_test.npy'), y_test)

    print("已保存 .npy 数据至:", save_dir)

if __name__ == "__main__":
    save_numpy_cifar10()
