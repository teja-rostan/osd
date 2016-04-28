import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
import load_cifar
from ionmf.factorization.onmf import onmf
import roc_auc as auc
import time
import pandas as pd

srng = RandomStreams()


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))


def rectify(X):
    return T.maximum(X, 0.)


def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')


def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X


# def fusion_update(weight_matrix, rank):
#     weights = weight_matrix.get_value()
#     (n_kernels, n_channels, w, h) = weights.shape
#     C = np.zeros((n_kernels, n_channels, w, h))
#     deg = []
#     for i in range(n_kernels):
#         d = 0
#         for j in range(n_channels):
#             W, H = onmf(weights[i, j], rank=rank, alpha=1.0)
#             A = W.dot(H)
#             threshold = np.mean(np.percentile(abs(A) - abs(weights[i, j]), 20))
#             c = (abs(A) - (abs(weights[i, j]))) < threshold
#             weights[i, j] *= c
#             C[i, j] = c
#             d += 1 - (np.count_nonzero(c) / len(c.flat))
#         deg.append(np.mean(d))
#     deg = np.mean(deg)
#     weight_matrix.set_value(weights)
#     return deg, C


# def repair(weight_matrix, c):
#     (n_kernels, n_channels, w, h) = c.shape
#     weights = weight_matrix.get_value()
#     for i in range(n_kernels):
#         for j in range(n_channels):
#             weights[i, j] *= c[i, j]
#     weight_matrix.set_value(weights)
#

def fusion_update(weight_matrix, rank):
    weights = weight_matrix.get_value()
    (n_kernels, n_channels, w, h) = weights.shape
    deg = []
    C = []
    for i in range(n_kernels):
        channels = [np.asarray(weights[i, j]).reshape(w*h, 1) for j in range(n_channels)]
        channels = np.hstack(np.asarray(channels))
        W, H = onmf(channels, rank=rank, alpha=1.0)
        A = W.dot(H)
        diff = abs(A) - abs(channels)
        threshold = np.mean(np.percentile(diff, 20))
        c = diff < threshold
        channels *= c
        C.append(c)
        d = 1 - (np.count_nonzero(c) / len(c.flat))
        deg.append(np.mean(d))
        vector_w = np.hsplit(channels, n_channels)
        for j in range(n_channels):
            weights[i, j] = vector_w[j].reshape(w, h)
    deg = np.mean(deg)
    weight_matrix.set_value(weights)
    return deg, C


def repair(weight_matrix, C):
    weights = weight_matrix.get_value()
    (n_kernels, n_channels, w, h) = weights.shape
    for i in range(n_kernels):
        channels = [np.asarray(weights[i, j]).reshape(w*h, 1) for j in range(n_channels)]
        channels = np.hstack(np.asarray(channels))
        channels *= C[i]
        vector_w = np.hsplit(channels, n_channels)
        for j in range(n_channels):
            weights[i, j] = vector_w[j].reshape(w, h)
    weight_matrix.set_value(weights)


def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates


def model(X, w, w2, w3, w4, p_drop_conv, p_drop_hidden):

    l1a = rectify(conv2d(X, w, border_mode='full'))
    l1 = max_pool_2d(l1a, (2, 2))
    l1 = dropout(l1, p_drop_conv)

    l2a = rectify(conv2d(l1, w2))
    l2 = max_pool_2d(l2a, (2, 2))
    l2 = dropout(l2, p_drop_conv)

    l3a = rectify(conv2d(l2, w3))
    l3b = max_pool_2d(l3a, (2, 2))
    l3 = T.flatten(l3b, outdim=2)
    l3 = dropout(l3, p_drop_conv)

    l4 = rectify(T.dot(l3, w4))
    l4 = dropout(l4, p_drop_hidden)

    pyx = softmax(T.dot(l4, w_o))
    return l1, l2, l3, l4, pyx


(trX, trY), (teX, teY), num_of_class = load_cifar.load_data()


X = T.ftensor4()
Y = T.fmatrix()

w = init_weights((32, 3, 5, 5))  # conv weights (n_kernels, n_channels, kernel_w, kernel_h)
w2 = init_weights((64, 32, 5, 5))
w3 = init_weights((128, 64, 5, 5))
w4 = init_weights((128 * 2 * 2, 625))  # highest conv layer has 128 filters and a 3x3 grid of responses
w_o = init_weights((625, 10))

noise_l1, noise_l2, noise_l3, noise_l4, noise_py_x = model(X, w, w2, w3, w4, 0.2, 0.5)
l1, l2, l3, l4, py_x = model(X, w, w2, w3, w4, 0., 0.)
y_x = T.argmax(py_x, axis=1)


cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
params = [w, w2, w3, w4, w_o]
params2 = [w, w2, w3]
updates = RMSprop(cost, params, lr=0.001)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

AUC = []
TIME = []
for i in range(30):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        cost = train(trX[start:end], trY[start:end])
    start = time.time()
    y_score = predict(teX)
    end = time.time()-start
    auc_r = auc.roc_auc(np.argmax(teY, axis=1), y_score)
    TIME.append(end)
    AUC.append(auc_r)
    print("AUC:", auc_r, "TIME:", end)

C = []
deg = []
for p in params2:
    d, c = fusion_update(p, 5)
    C.append(c)
    deg.append(d)
start = time.time()
y_score = predict(teX)
end = time.time()-start
auc_r = auc.roc_auc(np.argmax(teY, axis=1), y_score)
TIME.append(end)
AUC.append(auc_r)
print("MF AUC:", auc_r, "deg:", np.mean(deg), "TIME:", end)  # , "thresh:", np.mean(thresh)

for i in range(20):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        cost = train(trX[start:end], trY[start:end])
        for j in range(len(params2)):
            repair(params2[j], C[j])
    start = time.time()
    y_score = predict(teX)
    end = time.time()-start
    auc_r = auc.roc_auc(np.argmax(teY, axis=1), y_score)
    TIME.append(end)
    AUC.append(auc_r)
    print("AUC:", auc_r, "TIME:", end)

data = pd.DataFrame({"AUC": AUC, "TIME": TIME})
data.to_csv("cnnet_fusion_cifar_stacked2.csv", index=False)
