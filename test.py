# test.py

import yaml
import pickle
from model import MLP
from data_loader import load_cifar10
from loss_fn import compute_accuracy

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def test_model():
    _, _, _, _, X_test, y_test = load_cifar10(**config['data'])
    model = MLP(**config['model'])

    with open(config['train']['save_path'], 'rb') as f:
        model.params = pickle.load(f)

    logits, _ = model.forward(X_test)
    acc = compute_accuracy(logits, y_test)
    print(f"Test Accuracy: {acc:.2f}%")

if __name__ == '__main__':
    test_model()
