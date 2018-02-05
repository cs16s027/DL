import numpy as np

def activation_function(x, activation=activation):
    if (activation == 'sigmoid'):
        return 1/(1+np.exp(-x))
    elif (activation == 'tanh'):
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    else:
        return x

def output_function(x, activation='softmax'):
    if (activation == 'softmax'):
        x = np.exp(x - np.max(x))  # Normalization for numerical stability, from CS231n notes
        return x/np.sum(x, axis=0)
    if (activation == 'sigmoid'):
        return 1/(1+np.exp(-x))
    else:
        return x

def loss(y_true, y_pred, loss=loss):
    batch_size = y_true.shape[0]
    if loss == 'sq':
        return (1./(2*batch_size)) * np.sum((y_true - y_pred)**2)
    if loss == 'ce':
        return (-1./batch_size) * np.sum*(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))

