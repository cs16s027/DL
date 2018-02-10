import numpy as np

def gradient_descent(network, x, y, lr):
    y_pred, loss = network.forward(x, y)
    network.backward(y, y_pred)
    network.theta[:] = network.theta - lr * network.grad_theta
    return loss

