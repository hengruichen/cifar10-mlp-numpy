import numpy as np

def compute_loss(logits, labels, reg, params):
    m = logits.shape[0]
    exp_scores = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    one_hot = np.eye(probs.shape[1])[labels]
    log_loss = -np.mean(np.sum(one_hot * np.log(probs + 1e-8), axis=1))
    l2 = 0.5 * reg * sum(np.sum(params[k]**2) for k in params if 'W' in k)
    return log_loss + l2

def compute_accuracy(logits, labels):
    preds = np.argmax(logits, axis=1)
    return 100.0 * np.mean(preds == labels)
