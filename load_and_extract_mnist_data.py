import numpy as np
import pickle as pickle
from urllib.request import urlretrieve
import os
import gzip
def load_dataset():
    url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
    filename = 'mnist.pkl.gz'
    if not os.path.exists(filename):
        print("Downloading MNIST dataset...")
        urlretrieve(url, filename)
    with gzip.open(filename, 'rb') as f:
        data = pickle.load(f,encoding='bytes')
    X_train, y_train = data[0]
    X_val, y_val = data[1]
    X_test, y_test = data[2]
    X_train = X_train.reshape((-1, 1, 28, 28))
    X_val = X_val.reshape((-1, 1, 28, 28))
    X_test = X_test.reshape((-1, 1, 28, 28))
    y_train = y_train.astype(np.uint8)
    y_val = y_val.astype(np.uint8)
    y_test = y_test.astype(np.uint8)
    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__=="__main__":
    ### database format
    ### img: [num of img,img channel,img height,img weight]; label :[num,1]
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    first_train_img = X_train[0][0]
    first_train_label = y_train[0]
    print(first_train_img.shape)  ##should be 28*28
