import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
import load_cifar
# from skfusion import fusion
from ionmf.factorization.onmf import onmf
import roc_auc as auc
import time

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


def fusion_update(weight_matrix, rank):
    # define our data fusion graph
    # filtre sestevamo (po celicah v mrezi)
    # na diagonalo postavimo posamezne filtre
    # v preostale koordinate (recimo na i, j)
    #  bi postavili vsoto filtra i in filtra j
    #   0  1    2     3     4
    # 0    w0   w0+w1 w0+w2 w0+w3
    # 1         w1    w1+w2 w1+w3
    # 2               w2    w2+w3
    # 3                     w3
    # 4
    weights = weight_matrix.get_value()
    # print(weights.shape)
    (n_kernels, n_channels, w, h) = weights.shape
    C = np.zeros((n_kernels, n_channels, w, h))
    deg = []
    for i in range(n_kernels):
        d = 0
        for j in range(n_channels):
            W, H = onmf(weights[i, j], rank=rank, alpha=1.0)
            A = W.dot(H)
            # print(W.shape, H.shape, weights[i, j].shape, C.shape, w, h)
            threshold = np.mean(np.percentile(abs(A) - abs(weights[i, j]), 20))
            c = (abs(A) - (abs(weights[i, j]))) < threshold
            weights[i, j] *= c
            C[i, j] = c
            d += 1 - (np.count_nonzero(c) / len(c.flat))
        deg.append(np.mean(d))
    deg = np.mean(deg)
    weight_matrix.set_value(weights)
    return deg, C
    # t = [fusion.ObjectType('kernel_'+str(b), rank) for b in range(0, len(weights)+1)]
    # relations = []
    # indexes = []
    # indexes2 = []
    # for i in range(0, len(weights)):
    #     for j in range(0, len(weights)):
    #         if i == j:
    #             indexes.append(len(relations))
    #             indexes2.append(i)
    #             relations.append(fusion.Relation(weights[i], t[i], t[j+1]))
    #         elif i > j:
    #             relations.append(fusion.Relation(weights[i]+weights[j], t[i], t[j+1]))
    #
    # fusion_graph = fusion.FusionGraph()
    # fusion_graph.add_relations_from(relations)
    # # infer the latent data model
    # fuser = fusion.Dfmf()
    # fuser.fuse(fusion_graph)
    # c = np.zeros(len(weights))
    # print indexes
    # threshold = np.mean(np.percentile(abs(fuser.complete(relations)) - abs(w.get_value()), 20))
    # for k, idx in enumerate(indexes):
    #     c[k] = (abs(fuser.complete(relations[idx])) - (abs(w.get_value()[a, indexes2[i]]))) < threshold
    #     w[i, k] *= c[k]
    # all_c.append(c)


def repair(weight_matrix, c):
    (n_kernels, n_channels, w, h) = c.shape
    weights = weight_matrix.get_value()
    for i in range(n_kernels):
        for j in range(n_channels):
            weights[i, j] *= c[i, j]
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


# trX = trX.reshape(-1, 1, 32, 32)
# teX = teX.reshape(-1, 1, 32, 32)

X = T.ftensor4()
Y = T.fmatrix()

w = init_weights((32, 3, 3, 3))  # conv weights (n_kernels, n_channels, kernel_w, kernel_h)
w2 = init_weights((64, 32, 3, 3))
w3 = init_weights((128, 64, 3, 3))
w4 = init_weights((128 * 3 * 3, 625))  # highest conv layer has 128 filters and a 3x3 grid of responses
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

for i in range(30):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        cost = train(trX[start:end], trY[start:end])
    start = time.time()
    y_score = predict(teX)
    end = time.time()-start
    print("AUC:", auc.roc_auc(np.argmax(teY, axis=1), y_score), "TIME:", end)
C = []
deg = []
for p in params2:
    d, c = fusion_update(p, 5)
    C.append(c)
    deg.append(d)
start = time.time()
y_score = predict(teX)
end = time.time()-start
print("MF AUC:", auc.roc_auc(np.argmax(teY, axis=1), y_score), "deg:", np.mean(deg), "TIME:", end)  # , "thresh:", np.mean(thresh)
for i in range(15):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        cost = train(trX[start:end], trY[start:end])
        for j in range(len(params2)):
            repair(params2[j], C[j])
    start = time.time()
    y_score = predict(teX)
    end = time.time()-start
    print("AUC:", auc.roc_auc(np.argmax(teY, axis=1), y_score), "TIME:", end)