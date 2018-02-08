import numpy as np
from helpers import activation_function, output_activation, loss_function

class Network:

    def __init__(self, num_hidden, sizes, activation = 'softmax', loss = 'ce'):
        # L hidden layers, layer 0 is input, layer (L+1) is output
        sizes = [784] + sizes + [10]
        self.L = num_hidden
        # Parameter map from theta to Ws and bs
        self.param_map = {}
        start, end = 0, 0
        for i in range(1, L + 2):
            end = start + sizes[i - 1] * sizes[i]
            self.param_map['W{}'.format(i)] = (start, end)
            start = end
            end = start + sizes[i]
            self.param_map['b{}'.format(i)] = (start, end)
            start = end
        num_params = end
        # Allocate params
        self.theta = np.random.uniform(size = num_params)
        self.grad_theta = np.zeros((size = num_params))
        self.params = {}
        self.grad_params = {}
        for i in range(1, L + 2):
            weight = 'W{}'.format(i)
            start, end = self.param_map[weight]
            self.params[weight] = self.theta[start : end].reshape((sizes[i - 1], sizes[i]))
            self.grad_params[weight] = self.grad_theta[start : end].reshape((sizes[i - 1], sizes[i]))
            bias = 'b{}'.format(i)
            start, end = self.param_map[bias]
            self.params[bias] = self.theta[start : end].reshape((sizes[i], 1))
            self.grad_params[bias] = self.grad_theta[start : end].reshape((sizes[i], 1))

        self.activation = activation
        self.loss = loss

    def forward(self, x, y):
        # a(i) = b(i) + W(i)*h(i-1)
        # h(i) = g(i-1)
        self.activations = {}
        self.activations['h0'] = x
        for i in range (1, L + 1):
            self.activations['a{}'.format(i)] = self.params['b{}'.format(i)] + np.matmul(self.params['W{}'.format(i)], self.activations['h{}'.format(i-1)])
            self.activations['h{}'.format(i)] = activation_function(self.activations['a{}'.format(i)], self.activation)
        self.activations['a{}'.format(L + 1)] = self.params['b{}'.format(L + 1)] + np.matmul(self.params['W{}'.format(L+1)], self.activations['h{}'.format(L)])
        y_pred = output_function(self.activations['a{}'.format(L + 1)])
        loss = loss_function(y, y_pred, self.loss)
        return y_pred, loss

    def backward(y_true, y_pred):
        grad_activations = {}
        # Compute output gradient
        e_y = np.zeros(shape=y_pred.shape)
        e_y[np.argmax(y_true)] = 1
        grad_activations['a{}'.format(L + 1)] = -(e_y - y_pred)
        for k in range (L + 1, 0, -1):
            # Gradients wrt parameters
            self.grad_params['W{}'.format(k)] = np.matmul(grad_activations['a{}'.format(k)], self.activations['h{}'.format(k-1)].T)
            self.grad_params['b{}'.format(k)] = grad_activations['a{}'.format(k)]
            # Gradients wrt prev layer
            grad_activations['h{}'.format(k-1)] = mp.matmul(self.params['W{}'.format(k)].T, grad_activations['a{}'.format(k)])
            # Gradients wrt prev preactivation
            grad_activations['a{}'.format(k-1)] = np.multiply(grad_activations['h{}'.format(k-1)], activation_function(self.activations['a{}'.format(k-1)], self.activation))

