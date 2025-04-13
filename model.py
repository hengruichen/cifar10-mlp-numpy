import numpy as np
import pickle

# ---------- Activation Functions ----------

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(np.float32)

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

activation_funcs = {
    'relu': (relu, relu_derivative),
    'sigmoid': (sigmoid, sigmoid_derivative),
    'tanh': (tanh, tanh_derivative),
}

# ---------- MLP with 3 Hidden Layers ----------

def init_weight(shape, method='he'):
    fan_in = shape[0]
    fan_avg = (shape[0] + shape[1]) / 2
    if method == 'xavier':
        return np.random.randn(*shape) * np.sqrt(1.0 / fan_avg)
    elif method == 'he':
        return np.random.randn(*shape) * np.sqrt(2.0 / fan_in)
    elif method == 'normal':
        return np.random.randn(*shape) * 0.01
    else:
        raise ValueError(f"Unknown init method: {method}")

class MLP:
    def __init__(self, input_dim=3072, hidden_dim1=256, hidden_dim2=128, hidden_dim3=64,
                 output_dim=10, activation='relu', init_method='he'):
        assert activation in activation_funcs
        self.act, self.act_deriv = activation_funcs[activation]

        self.params = {
            'W1': init_weight((input_dim, hidden_dim1), method=init_method),
            'b1': np.zeros((1, hidden_dim1)),
            'W2': init_weight((hidden_dim1, hidden_dim2), method=init_method),
            'b2': np.zeros((1, hidden_dim2)),
            'W3': init_weight((hidden_dim2, hidden_dim3), method=init_method),
            'b3': np.zeros((1, hidden_dim3)),
            'W4': init_weight((hidden_dim3, output_dim), method=init_method),
            'b4': np.zeros((1, output_dim)),
        }

    def forward(self, X):
        Z1 = X @ self.params['W1'] + self.params['b1']
        A1 = self.act(Z1)
        Z2 = A1 @ self.params['W2'] + self.params['b2']
        A2 = self.act(Z2)
        Z3 = A2 @ self.params['W3'] + self.params['b3']
        A3 = self.act(Z3)
        Z4 = A3 @ self.params['W4'] + self.params['b4']
        cache = (X, Z1, A1, Z2, A2, Z3, A3, Z4)
        return Z4, cache

    def backward(self, logits, labels, cache, reg):
        X, Z1, A1, Z2, A2, Z3, A3, Z4 = cache
        m = X.shape[0]

        exp_scores = np.exp(Z4 - np.max(Z4, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        one_hot = np.eye(probs.shape[1])[labels]
        loss = -np.mean(np.sum(one_hot * np.log(probs + 1e-8), axis=1))
        loss += 0.5 * reg * sum(np.sum(self.params[k] ** 2) for k in self.params if 'W' in k)

        dZ4 = (probs - one_hot) / m
        dW4 = A3.T @ dZ4 + reg * self.params['W4']
        db4 = np.sum(dZ4, axis=0, keepdims=True)

        dA3 = dZ4 @ self.params['W4'].T
        dZ3 = dA3 * self.act_deriv(Z3)
        dW3 = A2.T @ dZ3 + reg * self.params['W3']
        db3 = np.sum(dZ3, axis=0, keepdims=True)

        dA2 = dZ3 @ self.params['W3'].T
        dZ2 = dA2 * self.act_deriv(Z2)
        dW2 = A1.T @ dZ2 + reg * self.params['W2']
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ self.params['W2'].T
        dZ1 = dA1 * self.act_deriv(Z1)
        dW1 = X.T @ dZ1 + reg * self.params['W1']
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        grads = {
            'W1': dW1, 'b1': db1,
            'W2': dW2, 'b2': db2,
            'W3': dW3, 'b3': db3,
            'W4': dW4, 'b4': db4,
        }
        return loss, grads

    def update_params(self, grads, lr):
        for k in self.params:
            self.params[k] -= lr * grads[k]

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.params, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.params = pickle.load(f)
