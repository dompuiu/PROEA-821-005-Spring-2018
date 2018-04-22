import numpy as np
from random import randrange, shuffle


class AveragedPerceptron:
    def __init__(self, learning_rate=1):
        self.learning_rate = learning_rate

    def train(self, train, labels, epochs):
        w = np.array([randrange(-100, 100, 1) / 10000 for _ in range(len(train[0]))])
        u = np.array([0.0 for _ in range(len(train[0]))])
        c = 1

        for _ in range(epochs):
            [_, w, u, c] = self.train_one_epoch(train, labels, w, u, c)

        return u / c

    def train_one_epoch(self, train, labels, w, u, c):
        lr = self.learning_rate
        order = [i for i in range(len(train))]
        shuffle(order)
        updates_count = 0

        for i in order:
            x = np.array(train[i])
            y = labels[i]

            if (y * np.dot(x, w)) < 0:
                w += x * y * lr
                updates_count += 1

            u += w
            c += 1

        return [updates_count, u / c, u, c]

    @staticmethod
    def predict(x, w):
        x = np.array(x)
        if np.dot(x, w) < 0:
            return -1
        else:
            return 1
