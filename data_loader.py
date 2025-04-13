import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

def load_images_from_folder(folder, label_map):
    X, y = [], []
    for label_name in sorted(os.listdir(folder)):
        class_folder = os.path.join(folder, label_name)
        if not os.path.isdir(class_folder): continue
        label = label_map[label_name]
        for filename in os.listdir(class_folder):
            img_path = os.path.join(class_folder, filename)
            img = Image.open(img_path).convert("RGB").resize((32, 32))
            X.append(np.asarray(img, dtype=np.uint8))
            y.append(label)
    return np.array(X), np.array(y)

def load_cifar10(data_dir='cifar10', val_ratio=0.1, shuffle=True, random_seed=42):
    train_path = os.path.join(data_dir, 'train')
    test_path = os.path.join(data_dir, 'test')

    label_names = sorted(os.listdir(train_path))
    label_map = {name: idx for idx, name in enumerate(label_names)}

    X_train_full, y_train_full = load_images_from_folder(train_path, label_map)
    X_test, y_test = load_images_from_folder(test_path, label_map)

    # Normalize and reshape
    X_train_full = X_train_full.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    X_train_full = X_train_full.reshape((X_train_full.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))

    # Split train into train + val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_ratio,
        shuffle=shuffle, random_state=random_seed, stratify=y_train_full
    )

    return X_train, y_train, X_val, y_val, X_test, y_test
