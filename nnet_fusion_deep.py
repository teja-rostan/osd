import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from skfusion import fusion
import load_cifar
import load_mnist
import roc_auc as auc

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


def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X


def fusion_update(w_h, w_h2, w_h3, w_h4, w_o, rank):
    t1 = fusion.ObjectType('Type 1', rank)
    t2 = fusion.ObjectType('Type 2', rank)
    t3 = fusion.ObjectType('Type 3', rank)
    t4 = fusion.ObjectType('Type 4', rank)
    t5 = fusion.ObjectType('Type 5', rank)
    t6 = fusion.ObjectType('Type 6', rank)
    relations = [fusion.Relation(w_h.get_value(), t1, t2),
                 fusion.Relation(w_h2.get_value(), t2, t3),
                 fusion.Relation(w_h3.get_value(), t3, t4),
                 fusion.Relation(w_h4.get_value(), t4, t5),
                 fusion.Relation(w_o.get_value(), t5, t6)]
    fusion_graph = fusion.FusionGraph()
    fusion_graph.add_relations_from(relations)
    fuser = fusion.Dfmf()
    fuser.fuse(fusion_graph)
    threshold = np.mean([np.percentile(abs(fuser.complete(relations[0])) - abs(w_h.get_value()), 10),
                         np.percentile(abs(fuser.complete(relations[1])) - abs(w_h2.get_value()), 10),
                         np.percentile(abs(fuser.complete(relations[2])) - abs(w_h3.get_value()), 10),
                         np.percentile(abs(fuser.complete(relations[3])) - abs(w_h4.get_value()), 10),
                         np.percentile(abs(fuser.complete(relations[4])) - abs(w_o.get_value()), 10)])
    c1 = (abs(fuser.complete(relations[0])) - (abs(w_h.get_value()))) < threshold
    c2 = (abs(fuser.complete(relations[1])) - (abs(w_h2.get_value()))) < threshold
    c3 = (abs(fuser.complete(relations[2])) - (abs(w_h3.get_value()))) < threshold
    c4 = (abs(fuser.complete(relations[3])) - (abs(w_h3.get_value()))) < threshold
    c5 = (abs(fuser.complete(relations[4])) - (abs(w_o.get_value()))) < threshold
    deg = 1 - (np.count_nonzero(c1) + np.count_nonzero(c2) + np.count_nonzero(c3) + np.count_nonzero(
            c4) + np.count_nonzero(c5)) / float(
        len(c1.flat) + len(c2.flat) + len(c3.flat) + len(c4.flat) + len(c5.flat))

    w_h.set_value(w_h.get_value() * c1)
    w_h2.set_value(w_h2.get_value() * c2)
    w_h3.set_value(w_h3.get_value() * c3)
    w_h4.set_value(w_h4.get_value() * c4)
    w_o.set_value(w_o.get_value() * c5)
    return deg, threshold, c1, c2, c3, c4, c5


def repair(w_h, w_h2, w_h3, w_h4, w_o, c1, c2, c3, c4, c5):
    w_h.set_value(w_h.get_value()*c1)
    w_h2.set_value(w_h2.get_value()*c2)
    w_h3.set_value(w_h3.get_value()*c3)
    w_h4.set_value(w_h4.get_value()*c4)
    w_o.set_value(w_o.get_value()*c5)


def model(X, w_h, w_h2, w_h3, w_h4, w_o, p_drop_input, p_drop_hidden):
    X = dropout(X, p_drop_input)
    h = rectify(T.dot(X, w_h))

    h = dropout(h, p_drop_hidden)
    h2 = rectify(T.dot(h, w_h2))

    h2 = dropout(h2, p_drop_hidden)
    h3 = rectify(T.dot(h2, w_h3))

    h3 = dropout(h3, p_drop_hidden)
    h4 = rectify(T.dot(h3, w_h4))

    h4 = dropout(h4, p_drop_hidden)
    py_x = softmax(T.dot(h4, w_o))
    return h, h2, h3, h4, py_x


(trX, trY), (teX, teY), num_of_class = load_mnist.load_data()
row, col = np.shape(trX)
# col = c * x * y
rank = num_of_class / 2
hidden = (col + num_of_class) * 2 / 3
print "Num of hidden neurons:", hidden
print "Num of classes:", num_of_class
print "Rank:", rank
X = T.fmatrix()
Y = T.fmatrix()

w_h = init_weights((col, hidden))  # 784 <- input
w_h2 = init_weights((hidden, hidden))
w_h3 = init_weights((hidden, hidden))
w_h4 = init_weights((hidden, hidden))
w_o = init_weights((hidden, num_of_class))  # 10 <- output

noise_h, noise_h2, noise_h3, noise_h4, noise_py_x = model(X, w_h, w_h2, w_h3, w_h4, w_o, 0.2, 0.5)
h, h2, h3, h4, py_x = model(X, w_h, w_h2, w_h3, w_h4, w_o, 0., 0.)
y_x = T.argmax(py_x, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
params = [w_h, w_h2, w_h3, w_h4, w_o]
updates = RMSprop(cost, params, lr=0.001)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

for i in range(100):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)): # 0,128; 128,256; ...
        cost = train(trX[start:end], trY[start:end])
    y_score = predict(teX)
    print "AUC:", auc.roc_auc(np.argmax(teY, axis=1), y_score)

deg, thresh, c1, c2, c3, c4, c5 = fusion_update(w_h, w_h2, w_h3, w_h4, w_o, rank)
y_score = predict(teX)
print "MF AUC:", auc.roc_auc(np.argmax(teY, axis=1), y_score), "deg:", np.mean(deg), "thresh:", np.mean(thresh)

for i in range(50):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        cost = train(trX[start:end], trY[start:end])
        repair(w_h, w_h2, w_h3, w_h4, w_o, c1, c2, c3, c4, c5)
    y_score = predict(teX)
    print "AUC:", auc.roc_auc(np.argmax(teY, axis=1), y_score)
