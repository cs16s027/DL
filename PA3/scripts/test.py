import sys, os
import shutil
import resource
import argparse
import logging
from helpers import setup_logger

import numpy as np
import tensorflow as tf

from data import loadData
from network import CNN
from augment import Augment
from modelparser import loadArch

def memory_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 / 2, hard))

def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory

# Test model on data after every epoch
def testInBatches(model, x, y, batch_size):
    num_batches = x.shape[0] / batch_size
    predictions = np.zeros((x.shape[0]))
    for batch in range(num_batches):
        start, end = batch * batch_size, (batch + 1) * batch_size
        batch_x, batch_y = x[start : end], y[start : end]
        predictions[start : end] = model.getPredictions(batch_x, batch_y)[0]
    return predictions

def writePredictions(predictions, results_file):
    with open(results_file, 'wb') as f:
        f.write('id,label\n')
        for index, p in enumerate(predictions):
            f.write('{},{}\n'.format(index, int(p)))

# Set memory limit
#memory_limit()

### Test Model
# Parse args
parser = argparse.ArgumentParser(description='Train the CNN')
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
parser.add_argument('--arch', default='models/cnn.json',
                    help = 'path to model architecture')
parser.add_argument('--model_name', default = 'model',
                    help = 'name of the model to save logs, weights')
parser.add_argument('--lr', default = 0.001,
                    help = 'learning rate')
parser.add_argument('--init', default = '1',
                    help = 'initialization')
parser.add_argument('--batch_size', default = 20,
                    help = 'batch_size')
args = parser.parse_args()

# Load data
train_path, valid_path, test_path = args.train, args.val, args.test
logs_path = args.expt_dir
model_path, model_arch, model_name = args.save_dir, args.arch, args.model_name
model_path = os.path.join(model_path, model_name)
lr, batch_size, init = float(args.lr), int(args.batch_size), int(args.init)

data = loadData(train_path, valid_path, test_path)
train_X, train_Y, valid_X, valid_Y, test_X, test_Y = data['train']['X'], data['train']['Y'],\
                                                     data['valid']['X'], data['valid']['Y'],\
                                                     data['test']['X'], data['test']['Y'],

# Load architecture
arch = loadArch(model_arch)

# GPU config
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.5)
with tf.Session(config = tf.ConfigProto(gpu_options=gpu_options)) as session:
    np.random.seed(100)
    model = CNN(arch, session, logs_path, init, lr)
    conv1_before = session.run([model.params['Wc1']])
    model.load(model_path)
    conv1_after = session.run([model.params['Wc1']])
    predictions = testInBatches(model, test_X, test_Y, batch_size)
    results_file = os.path.join(model_path, 'results.txt')
    writePredictions(predictions, results_file)

