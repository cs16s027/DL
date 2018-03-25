import sys, os
import shutil
import resource
import argparse
import logging
from helpers import setup_logger

import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Conv2D, BatchNormalization, Input, Activation, Flatten
from keras.optimizers import Adam

from data import loadData
from augment import Augment

def getModel(lr = 0.001):
    input_img = Input(shape = (1, 28, 28), name = 'X')
    x = Conv2D(8, (3, 3), strides = (2, 2), data_format = 'channels_first',\
                    padding = 'same', name = 'conv1')(input_img)
    x = Conv2D(16, (3, 3), strides = (2, 2), data_format = 'channels_first',\
                    padding = 'same', name = 'conv2')(x)
    x = Activation('relu')(x)
    x = Flatten(name = 'flatten')(x)
    x = Dense(512, activation = 'relu', name = 'fc_1')(x)
    x = Dense(128, activation = 'relu', name = 'fc_3')(x)
    y = Dense(10, activation = 'softmax', name = 'Y')(x)
    model = Model(input_img, y)

    model.compile(loss='categorical_crossentropy',\
                  optimizer=Adam(lr = lr), metrics = ['accuracy'])
    print model.summary()
    return model

def trainModel():
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
    model_path, model_name = args.save_dir, args.model_name
    model_path = os.path.join(model_path, model_name)
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    lr, batch_size, init = float(args.lr), int(args.batch_size), int(args.init)

    data = loadData(train_path, valid_path, test_path)
    train_X, train_Y, valid_X, valid_Y, test_X, test_Y = data['train']['X'], data['train']['Y'],\
                                                         data['valid']['X'], data['valid']['Y'],\
                                                         data['test']['X'], data['test']['Y'],


    # Logging
    train_log_name = '{}.train.log'.format(model_name)
    valid_log_name = '{}.valid.log'.format(model_name)
    train_log = setup_logger('train-log', os.path.join(logs_path, train_log_name))
    valid_log = setup_logger('valid-log', os.path.join(logs_path, valid_log_name))

    # Train
    num_epochs = 500
    num_batches = int(float(train_X.shape[0]) / batch_size)
    steps = 0
    patience = 100
    early_stop=0

    model = getModel(lr)
    loss_history = [np.inf]
    for epoch in range(num_epochs):
        print 'Epoch {}'.format(epoch)
        steps = 0
        indices = np.arange(train_X.shape[0])
        np.random.shuffle(indices)
        train_X, train_Y = train_X[indices], train_Y[indices]
        for batch in range(num_batches):
            start, end = batch * batch_size, (batch + 1) * batch_size
            x, y = Augment(train_X[range(start, end)]).batch, train_Y[range(start, end)]
            model.fit(x.reshape((-1, 1, 28, 28)), y, batch_size = batch_size, verbose = 0)
            steps += batch_size
            if steps % train_X.shape[0] == 0 and steps != 0:
                train_loss, train_acc = model.evaluate(train_X.reshape((-1, 1, 28, 28)), train_Y)
                train_log.info('Epoch {}, Step {}, Loss: {}, Accuracy: {}, lr: {}'.format(epoch, steps, train_loss, train_acc, lr))
                valid_loss, valid_acc = model.evaluate(valid_X.reshape((-1, 1, 28, 28)), valid_Y)
                valid_log.info('Epoch {}, Step {}, Loss: {}, Accuracy: {}, lr: {}'.format(epoch, steps, valid_loss, valid_acc, lr))
                if valid_loss < min(loss_history):
                    save_path = os.path.join(model_path, 'model')
                    model.save(save_path)
                    early_stop = 0
                early_stop += 1
                if (early_stop >= patience):
                    print "No improvement in validation loss for " + str(patience) + " steps - stopping training!"
                    print("Optimization Finished!")
                    return 1
                loss_history.append(valid_loss)
    print("Optimization Finished!")

if __name__ == '__main__':
    trainModel()
