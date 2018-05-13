import numpy as np
from random import shuffle
from scipy.sparse import csr_matrix


class SVM:
    def __init__(self, learning_rate=1, regularization_loss_tradeoff=1):
        self.learning_rate = learning_rate
        self.regularization_loss_tradeoff = regularization_loss_tradeoff

    def train(self, train, labels, epochs):
        w = csr_matrix((1, train[0].shape[1]), dtype=np.float128)

        for _ in range(epochs):
            [w] = self.train_one_epoch(train, labels, w)

        return w

    def train_one_epoch(self, train, labels, w):
        lr = self.learning_rate
        tradeoff = self.regularization_loss_tradeoff

        order = [i for i in range(train.shape[0])]
        shuffle(order)
        w_transpose = w.transpose()

        for i in order:
            x = train[i]
            y = labels.toarray()[0][i]

            if (x.dot(w_transpose) * y)[0, 0] <= 1:
                w = w * (1 - lr) + x * (lr * tradeoff * y)
            else:
                w = w * (1 - lr)

            w_transpose = w.transpose()

        return [w]


class SVMPredictor:
    def __init__(self, w):
        self.w = w.transpose()

    def predict(self, x):
        if x.dot(self.w)[0, 0] < 0:
            return -1
        else:
            return 1
