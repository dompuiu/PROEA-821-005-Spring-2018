import numpy as np
from scipy.sparse import csc_matrix


class LogisticRegression:
    @staticmethod
    def sigmoid(scores):
        return 1 / (1 + np.exp(-scores.toarray()))

    def __init__(self, learning_rate=1, tradeoff=1):
        self.learning_rate = learning_rate
        self.tradeoff = tradeoff

    def train(self, train, labels, epochs):
        w = csc_matrix((1, train[0].shape[1]), dtype=np.float128)

        for _ in range(epochs):
            [w] = self.train_one_epoch(train, labels, w)

        return w

    def train_one_epoch(self, train, labels, w):
        scores = train.dot(w.T)
        predictions = self.__class__.sigmoid(scores)

        # Update weights with gradient
        size = labels.shape[1]
        output_error_signal = labels - predictions.T
        gradient = train.T.dot(output_error_signal.T) / size
        w = w * (1 - self.learning_rate) + csc_matrix(gradient * self.learning_rate * self.tradeoff, dtype=np.float128).T

        return [w]


class LogisticRegressionPredictor:
    def __init__(self, w):
        self.w = w.transpose()

    def predict(self, x):
        return round(LogisticRegression.sigmoid(x.dot(self.w))[0][0])
