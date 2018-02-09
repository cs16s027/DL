import numpy as np
import h5py

def loadData(fname):
    lines = [line.strip().split(',') for line in open(fname, 'r').readlines()][1 : ]
    size = len(lines)
    dim = 784
    scale = 255.0
    X, Y = np.zeros((size, dim), dtype = np.float32), np.zeros((size), dtype = np.float32)
    for index, line in enumerate(lines):
        X[index][:] = np.float32(line[1 : -1]) / scale
        Y[index] = np.float32(line[-1])
    print 'Loaded data of shape', X.shape, Y.shape
    return X, Y

def normalizationParams(X):
    mean = X.mean(axis = 0)
    stddev = np.sqrt(np.mean((X - mean) * (X - mean), axis = 0))
    return mean, stddev

def prepareData():
    normalize = True
    train_X, train_Y = loadData('data/train.csv')
    valid_X, valid_Y = loadData('data/val.csv')
    
    if normalize == True:
        mean, stddev = normalizationParams(train_X)
        train_X = (train_X - mean) / stddev
        valid_X = (valid_X - mean) / stddev

    np.save('data/train_X.npy', train_X)
    np.save('data/train_Y.npy', train_Y)
    np.save('data/valid_X.npy', valid_X)
    np.save('data/valid_Y.npy', valid_Y)   

def loadData():
    train_X, train_Y = np.load('data/train_X.npy'), np.load('data/train_Y.npy')
    valid_X, valid_Y = np.load('data/valid_X.npy'), np.load('data/valid_Y.npy')
    return train_X, train_Y, valid_X, valid_Y

if __name__ == '__main__':

    pass
    


