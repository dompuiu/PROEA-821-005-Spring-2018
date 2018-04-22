import numpy as np
from random import randrange, shuffle


class SimplePerceptron:
    def __init__(self, learning_rate=1, epochs=20):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def train_weights(self, train, labels):
        w = np.array([randrange(-100, 100, 1) / 10000 for _ in range(len(train[0]))])
        lr = self.learning_rate
        order = [i for i in range(len(train))]

        for _ in range(self.epochs):
            shuffle(order)

            for i in order:
                x = np.array(train[i])
                y = labels[i]

                if (y * np.dot(x, w)) < 0:
                    w += x * y * lr

        return w

    @staticmethod
    def predict(x, w):
        x = np.array(x)
        if np.dot(x, w) < 0:
            return -1
        else:
            return 1
