import os, urllib, gzip, cPickle
import numpy as np


def one_hot(x, n):
    x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


def load_data():
    dataset = 'mnist.pkl.gz'
    origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == dataset:
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == dataset:
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print "Opening data"
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    trX, trY = train_set
    valX, valY = valid_set
    teX, teY = test_set
    num_of_class = len(np.unique(trY))
    trY = one_hot(trY, num_of_class)
    teY = one_hot(teY, num_of_class)
    print np.shape(trX)
    return (trX, trY), (teX, teY), num_of_class