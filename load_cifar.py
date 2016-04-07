import pickle
import numpy as np
import sys
import os
import urllib

def one_hot(x, n):
    x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


def load_batch(fpath, label_key='labels'):
    f = open(fpath, 'rb')
    if sys.version_info < (3,):
        d = pickle.load(f)
    else:
        d = pickle.load(f, encoding="bytes")
        # decode utf8
        for k, v in d.items():
            del(d[k])
            d[k.decode("utf8")] = v
    f.close()
    data = d["data"]
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels


def load_data():
    dataset = "cifar-10-batches-py"
    origin = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    path = "cifar/cifar-10-batches-py"
    data_dir, data_file = os.path.split(dataset)

    # if (not os.path.isfile(dataset)) and data_file == dataset:
    #     print 'Downloading data from %s' % origin
    #     urllib.urlretrieve(origin, dataset)

    nb_train_samples = 50000

    X_train = np.zeros((nb_train_samples, 3, 32, 32), dtype="uint8")
    y_train = np.zeros((nb_train_samples,), dtype="uint8")

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        data, labels = load_batch(fpath)
        X_train[(i-1)*10000:i*10000, :, :, :] = data
        y_train[(i-1)*10000:i*10000] = labels

    fpath = os.path.join(path, 'test_batch')

    X_test, y_test = load_batch(fpath)
    num_of_class = len(np.unique(y_train))
    # y_train = np.reshape(y_train, (len(y_train), 1))
    # y_test = np.reshape(y_test, (len(y_test), 1))
    y_train = one_hot(y_train, num_of_class)
    y_test = one_hot(y_test, num_of_class)
    return (X_train, y_train), (X_test, y_test), num_of_class