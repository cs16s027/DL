import sys, os
import argparse
import logging
from helpers import setup_logger

import numpy as np
import tensorflow as tf

from data import loadData
from network import CNN



# Parse args
parser = argparse.ArgumentParser(description='Train the MLP')
parser.add_argument('--expt_dir', default='./logs',
                    help='save dir for experiment logs')
parser.add_argument('--train', default='./data',
                    help='path to training set')
parser.add_argument('--val', default='./data',
                    help='path to validation set')
parser.add_argument('--test', default='./data',
                    help='path to test set')
parser.add_argument('--save_dir', default='./models',
                    help='path to save model')
args = parser.parse_args()

def parse_arch():
    pass

# Load data
train_path, valid_path, test_path = args.train, args.val, args.test
logs_path = args.expt_dir
model_path = args.save_dir
data = loadData(train_path, valid_path, test_path)
train_X, train_Y, valid_X, valid_Y, test_X, test_Y = data['train']['X'], data['train']['Y'],\
                                                     data['valid']['X'], data['valid']['Y'],\
                                                     data['test']['X'], data['test']['Y'],
# Logging
train_log_name = 'train.log'
valid_log_name = 'valid.log'
train_log = setup_logger('train-log', os.path.join(logs_path, train_log_name))
valid_log = setup_logger('valid-log', os.path.join(logs_path, valid_log_name))

# Train
num_epochs = 5
batch_size = 20
num_batches = int(float(train_X.shape[0]) / batch_size)
steps = 0

conv_sizes = [(5, 5, 1, 32), (5, 5, 32, 32)]
dense_sizes = [7 * 7 * 32, 1024]
num_out = 10
arch = ['input', 'conv', 'pool', 'conv', 'pool', 'dense', 'out']
model_name = 'cnn'
with tf.Graph().as_default(), tf.Session() as session:
    model = CNN(conv_sizes, dense_sizes, num_out, arch, session)
    loss_history = [np.inf]
    for epoch in range(num_epochs):
        steps = 0
        indices = np.arange(train_X.shape[0])
        np.random.shuffle(indices)
        train_X, train_Y = train_X[indices], train_Y[indices]
        for batch in range(num_batches):
            start, end = batch * batch_size, (batch + 1) * batch_size
            x, y = train_X[range(start, end)], train_Y[range(start, end)]
            model.step(x,y)
            steps += batch_size
            if steps % train_X.shape[0] == 0 and steps != 0:
                train_loss, train_acc = model.performance(train_X, train_Y)
                train_log.info('Epoch {}, Step {}, Loss: {}, Accuracy: {}, lr: {}'.format(epoch, steps, train_loss, train_acc, model.lr))
                valid_loss, valid_acc = model.performance(valid_X,valid_Y)
                valid_log.info('Epoch {}, Step {}, Loss: {}, Accuracy: {}, lr: {}'.format(epoch, steps, valid_loss, valid_acc, model.lr))
                if valid_loss < min(loss_history):
                    model.save(os.path.join(model_path, model_name))
                loss_history.append(valid_loss)
    print("Optimization Finished!")
