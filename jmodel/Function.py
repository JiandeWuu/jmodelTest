from .np import *

def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))

    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def accuracy(batch_y_pred, batch_y_val):
    count = 0
    for i in range(batch_y_pred.shape[0]):
        for j in range(batch_y_pred.shape[1]):
            if batch_y_pred[i, j] == batch_y_val[i, j]:
                count += 1
    return count / batch_y_pred.size