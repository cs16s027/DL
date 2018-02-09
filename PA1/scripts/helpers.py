import numpy as np

def activation_function(x, activation = 'sigmoid'):
    if (activation == 'sigmoid'):
        return 1 / (1 + np.exp(-x))
    elif (activation == 'tanh'):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    else:
        return x

def output_function(x, activation = 'softmax'):
    if (activation == 'softmax'):
        x = np.exp(x - np.max(x))  # Normalization for numerical stability, from CS231n notes
        return x/np.sum(x, axis=0)
    if (activation == 'sigmoid'):
        return 1 / (1 + np.exp(-x))
    else:
        return x

def loss_function(y_true, y_pred, loss = 'ce'):
    batch_size = y_true.shape[0]
    if loss == 'sq':
        return (1./(2*batch_size)) * np.sum((y_true - y_pred)**2)
    if loss == 'ce':
        return (-1.0 / batch_size) * np.log(y_pred[[int(index) for index in y_true], range(batch_size)]).sum()

