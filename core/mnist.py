import torch
import numpy as np
from keras.datasets import mnist

def load_mnist(p):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    (train_X, train_y_idx), (test_X, test_y_idx) = mnist.load_data()
    n = train_X.shape[0]
    nt = test_X.shape[0]
    train_X = torch.from_numpy(train_X).to(torch.float32).to(device)
    test_X = torch.from_numpy(test_X).to(torch.float32).to(device)

    train_y = np.zeros((n,10))
    for i in range(train_y.shape[0]):
        train_y[i,train_y_idx[i]] = 1.0
    train_y = torch.from_numpy(train_y).to(torch.float32).to(device)

    test_y = np.zeros((nt,10))
    for i in range(test_y.shape[0]):
        test_y[i,test_y_idx[i]] = 1.0
    test_y = torch.from_numpy(test_y).to(torch.float32).to(device)

    if p:
        print(device)
        print('X_train: ' + str(train_X.shape))
        print('Y_train: ' + str(train_y.shape))
        print('X_test:  '  + str(test_X.shape))
        print('Y_test:  '  + str(test_y.shape))

    return device,train_X, train_y, test_X, test_y