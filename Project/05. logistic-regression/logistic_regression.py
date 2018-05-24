import numpy as np
from random import randrange, shuffle


class LogisticRegression:
    @staticmethod
    def sigmoid(scores):
        return 1 / (1 + np.exp(-scores))

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

        scores = np.dot(train, w)
        predictions = self.__class__.sigmoid(scores)

        size = len(labels)
        output_error_signal = labels - predictions
        gradient = np.dot(np.array(train).T, output_error_signal) / size
        w = w * (1 - lr) + gradient * lr * tradeoff

        return [0, w]

    @staticmethod
    def predict(x, w):
        x = np.array(x)
        if np.dot(x, w) < 0:
            return -1
        else:
            return 1
