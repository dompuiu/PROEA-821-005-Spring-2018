import numpy as np
from random import randrange


class SimplePerceptron:
    def __init__(self, learning_rate=1):
        self.learning_rate = learning_rate

    def train_weights(self, train, labels):
        w = [randrange(-100, 100, 1) / 1000 for _ in range(len(train[0]))]
        # w = [0 for _ in range(len(train[0]))]

        for i, x in enumerate(train):
            x = np.array(x)
            y = labels[i]
            lr = self.learning_rate

            if ((np.dot(x, w)) * y) <= 0:
                w += x * y * lr

        return w

    @staticmethod
    def predict(x, w):
        x = np.array(x)
        if np.dot(x, w) <= 0:
            return -1
        else:
            return 1
