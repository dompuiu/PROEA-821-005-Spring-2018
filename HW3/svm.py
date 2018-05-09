from random import shuffle
from operations import vector_scalar_product, dot_product, vectors_sum


class SVM:
    def __init__(self, learning_rate=1):
        self.learning_rate = learning_rate

    def train(self, train, labels, epochs):
        w = {}
        lr = self.learning_rate

        for _ in range(epochs):
            order = [i for i in range(len(train))]
            shuffle(order)

            for i in order:
                x = train[i]
                y = labels[i]

                if (y * dot_product(x, w)) < 0:
                    w = vectors_sum(w, vector_scalar_product(x, y * lr))

        return w

    @staticmethod
    def predict(x, w):
        if dot_product(x, w) < 0:
            return -1
        else:
            return 1
