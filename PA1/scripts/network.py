import numpy as np
from helpers import activation_function, output_function, loss_function

class Network:

    def __init__(self, num_hidden, sizes, activation_choice = 'softmax', output_choice = 'softmax', loss_choice = 'ce'):
        # L hidden layers, layer 0 is input, layer (L+1) is output
        sizes = [784] + sizes + [10]
        self.L = num_hidden
        self.output_shape = 10
        # Parameter map from theta to Ws and bs
        self.param_map = {}
        start, end = 0, 0
        for i in range(1, self.L + 2):
            end = start + sizes[i - 1] * sizes[i]
            self.param_map['W{}'.format(i)] = (start, end)
            start = end
            end = start + sizes[i]
            self.param_map['b{}'.format(i)] = (start, end)
            start = end
        num_params = end
        # Allocate params
        self.theta = np.random.uniform(size = num_params)
        self.grad_theta = np.zeros((num_params))
        self.params = {}
        self.grad_params = {}
        for i in range(1, self.L + 2):
            weight = 'W{}'.format(i)
            start, end = self.param_map[weight]
            self.params[weight] = self.theta[start : end].reshape((sizes[i], sizes[i - 1]))
            self.grad_params[weight] = self.grad_theta[start : end].reshape((sizes[i], sizes[i - 1]))
            bias = 'b{}'.format(i)
            start, end = self.param_map[bias]
            self.params[bias] = self.theta[start : end].reshape((sizes[i], 1))
            self.grad_params[bias] = self.grad_theta[start : end].reshape((sizes[i], 1))

        self.activation_choice = activation_choice
        self.output_choice = output_choice
        self.loss_choice = loss_choice

    def forward(self, x, y):
        # a(i) = b(i) + W(i)*h(i-1)
        # h(i) = g(i-1)
        self.activations = {}
        self.activations['h0'] = x
        self.batch_size = x.shape[1]
        for i in range (1, self.L + 1):
            self.activations['a{}'.format(i)] = self.params['b{}'.format(i)] + np.matmul(self.params['W{}'.format(i)], self.activations['h{}'.format(i-1)])
            self.activations['h{}'.format(i)] = activation_function(self.activations['a{}'.format(i)], self.activation_choice)
        self.activations['a{}'.format(self.L + 1)] = self.params['b{}'.format(self.L + 1)] + np.matmul(self.params['W{}'.format(self.L+1)], self.activations['h{}'.format(self.L)])
        y_pred = output_function(self.activations['a{}'.format(self.L + 1)], self.output_choice)
        loss = loss_function(y, y_pred, self.loss_choice)
        return y_pred, loss

    def backward(self, y_true, y_pred):
        grad_activations = {}
        # Compute output gradient
        e_y = np.zeros((self.output_shape, self.batch_size))
        e_y[[int(index) for index in y_true], range(self.batch_size)] = 1
        grad_activations['a{}'.format(self.L + 1)] = -(e_y - y_pred)
        for k in range (self.L + 1, 0, -1):
            # Gradients wrt parameters
            self.grad_params['W{}'.format(k)] = np.matmul(grad_activations['a{}'.format(k)], self.activations['h{}'.format(k-1)].T)
            self.grad_params['b{}'.format(k)] = np.sum(grad_activations['a{}'.format(k)], axis = 1)
            # Do not compute gradients with respect to the inputs
            if k == 1:
                break
            # Gradients wrt prev layer
            grad_activations['h{}'.format(k-1)] = np.matmul(self.params['W{}'.format(k)].T, grad_activations['a{}'.format(k)])
            # Gradients wrt prev preactivation
            activation_ = activation_function(self.activations['a{}'.format(k - 1)], self.activation_choice)
            grad_activation_ = np.multiply(activation_, (1 - activation_))
            grad_activations['a{}'.format(k-1)] = np.multiply(grad_activations['h{}'.format(k-1)], grad_activation_)

        return self.grad_theta

