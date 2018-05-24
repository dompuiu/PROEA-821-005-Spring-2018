import numpy as np
from random import randrange, shuffle


class SVM:
    def __init__(self, learning_rate=1, regularization_loss_tradeoff=1):
        self.learning_rate = learning_rate
        self.regularization_loss_tradeoff = regularization_loss_tradeoff

    def train(self, train, labels, epochs):
        w = np.array([0 for _ in range(len(train[0]))])

        for _ in range(epochs):
            [_, w] = self.train_one_epoch(train, labels, w)

        return w

    def train_one_epoch(self, train, labels, w):
        lr = self.learning_rate
        tradeoff = self.regularization_loss_tradeoff

        order = [i for i in range(len(train))]
        shuffle(order)
        updates_count = 0

        for i in order:
            x = np.array(train[i])
            y = labels[i]

            if (y * np.dot(x, w)) <= 1:
                w = w * (1 - lr) + x * y * lr * tradeoff
                updates_count += 1
            else:
                w = w * (1 - lr)

        return [updates_count, w]

    @staticmethod
    def predict(x, w):
        x = np.array(x)
        if np.dot(x, w) < 0:
            return -1
        else:
            return 1
