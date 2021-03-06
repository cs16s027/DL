import sys, os
import tensorflow as tf
import numpy as np

def conv2d(x, W, b, strides=1, padding = 'SAME'):
    x = tf.nn.conv2d(x, W, strides = [1, strides, strides, 1], padding = padding)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def pool2d(x, filter_size, stride = 2, padding = 'SAME'):
    return tf.nn.max_pool(x, ksize=[1, filter_size, filter_size, 1], strides=[1, stride, stride, 1],
                          padding = padding)

def dense(x, W, b):
    fc = tf.add(tf.matmul(x, W), b)
    return tf.nn.relu(fc)

class CNN:

    def __init__(self, arch, session, logs_path, initializer = 1, lr = 0.001):
        self.params = {}
        self.layers = {}
        self.arch = arch
        self.lr = lr
        self.logs_path = logs_path
        # Initializers
        # He 
        if initializer == 2:
            init = tf.contrib.layers.variance_scaling_initializer()
        else:
            init = tf.contrib.layers.xavier_initializer()
        for item in self.arch:
            layer = item['name']
            params = item['params']
            if params == 'NONE':
                continue
            else:
                weight = params['weight']['name']
                shape = params['weight']['shape']
                self.params[weight] = tf.get_variable(name = weight, shape = shape, initializer = init)
                bias = params['bias']['name']
                shape = params['bias']['shape']
                self.params[bias] = tf.get_variable(name = bias, shape = shape, initializer = init)

        # Build the TensorFlow graph
        self.sess = session
        self.build_graph()

    def build_graph(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 784], name='input_data')
        self.y = tf.placeholder(tf.float32, shape=[None, 10], name='input_labels')
        self.keep_prob = tf.placeholder(tf.float32)
        self.prob = 1.0

        prev_layer = ''
        for index, item in enumerate(self.arch):
            layer = item['name']
            if item['params'] != 'NONE':
                params = item['params']
                weight = params['weight']['name']
                bias = params['bias']['name']

            if 'input' in layer:
                self.layers[layer] = tf.reshape(self.x, shape=[-1, 28, 28, 1])

            elif 'conv' in layer:
                padding, stride = item['padding'], item['stride']
                self.layers[layer] = conv2d(self.layers[prev_layer], self.params[weight], self.params[bias], stride, padding)

            elif 'pool' in layer:
                filter_size = item['filter_size']
                padding, stride = item['padding'], item['stride']
                self.layers[layer] = pool2d(self.layers[prev_layer], filter_size, stride, padding)

            elif 'reshape' in layer:
                shape_ = np.product(self.layers[prev_layer].get_shape().as_list()[1 : ])
                self.layers[layer] = tf.reshape(self.layers[prev_layer], [-1, shape_])

            elif 'dropout' in layer:
                self.prob = item['prob']
                self.layers[layer] = tf.nn.dropout(self.layers[prev_layer], self.keep_prob)

            elif 'batchnorm' in layer:
                self.layers[layer] = tf.contrib.layers.batch_norm(self.layers[prev_layer])

            elif 'fc' in layer:
                self.layers[layer] = dense(self.layers[prev_layer], self.params[weight], self.params[bias])

            elif 'output' in layer:
                logits = tf.add(tf.matmul(self.layers[prev_layer], self.params['Wout']), self.params['bout'])
                self.layers[layer] = logits
                y_pred =  tf.nn.softmax(logits)

            else:
                print "Invalid architecture!"
                sys.exit(1)

            print 'Adding Layer-{} : {}, Shape = {}'.format((index + 1), layer, self.layers[layer].get_shape())

            prev_layer = layer

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.y))
        optimizer = tf.train.AdamOptimizer(learning_rate = self.lr)
        self.train_op = optimizer.minimize(self.loss)
        self.predictions = tf.argmax(y_pred, 1)
        correct_pred = tf.equal(self.predictions, tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def step(self, batch_x, batch_y):
        self.sess.run(self.train_op, feed_dict = {self.x: batch_x, self.y: batch_y, self.keep_prob : self.prob})

    def performance(self, batch_x, batch_y):
        loss, acc = self.sess.run([self.loss, self.accuracy],
                                   feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob : 1.0})
        return loss, acc

    def getPredictions(self, batch_x, batch_y):
        return self.sess.run([self.predictions], 
                                     feed_dict = {self.x : batch_x, self.y : batch_y, self.keep_prob : 1.0}) 

    def save(self, save_path):
        saver = tf.train.Saver()
        saver.save(self.sess, save_path)

    def load(self, load_path):
        saver = tf.train.Saver()
        saver.restore(self.sess, os.path.join(load_path, 'model'))

