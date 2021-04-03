import os
import numpy as np
import urllib.request
import gzip
import pickle
import time

DATA_URL = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
MNIST_GZ_FILENAME = 'mnist.pkl.gz'


def __download_dataset(path, retries=4):
    try:
        urllib.request.urlretrieve(DATA_URL, MNIST_GZ_FILENAME)
        urllib.request.urlcleanup()

    except Exception as e:
        if retries:
            print(f'Retrying download due to error. Retries left: {retries}')
            time.sleep(4)
            __download_dataset(path, retries - 1)

        else:
            raise Exception(e)


def __scale(X):
    return (X - np.min(X)) / (np.max(X) - np.min(X))


def process_train_test(train_X, test_X, num_train_datapoints):
    # scale features to [0, 1]
    (train_X_scaled, test_X_scaled) = map(__scale, (train_X, test_X))

    # shuffle training set
    shuffled_idxs = np.random.shuffle(np.arange(train_X.shape[0]))
    train_X_shuffle = train_X_scaled[shuffled_idxs].squeeze(0)
    train_X_trunc = train_X_shuffle[:num_train_datapoints]

    return train_X_trunc, test_X_scaled


def load_processed_datasets(path=MNIST_GZ_FILENAME, num_train_datapoints=10**4):

    if not os.path.exists(path):
        print(f'Data not found at path: {path}. Downloading...')
        __download_dataset(path)

    with gzip.open(path, 'rb') as fgz:
        (train_X, _), (_, _), (test_X,
                               _) = pickle.load(fgz, encoding='latin1')
        
    return process_train_test(train_X, test_X, num_train_datapoints)
