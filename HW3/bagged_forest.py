import numpy as np
from scipy.sparse import csr_matrix
from decision_tree import DecisionTree


class BaggedForest:
    def __init__(self, trees_count):
        self.trees_count = trees_count

    def train(self, train, labels):
        trees = []

        for i in range(self.trees_count):
            if i > 0 and i % 100 == 0:
                print('Generating tree ' + str(i) + '...')

            random_indexes = np.round((train.shape[0] - 1) * np.random.random_sample(100))
            random_entries = train[random_indexes]
            random_labels = csr_matrix(labels).T[random_indexes]

            labels_description = ['Feature ' + str(i) for i in range(train.shape[1])]

            tree = DecisionTree(random_entries, random_labels, labels_description, 3).make_tree()
            trees.append(tree)

        print('Generated ' + str(i) + ' trees.')
        return trees


class BaggedForestPredictor:
    def __init__(self, trees):
        self.trees = trees

    def predict(self, x):
        labels_description = ['Feature ' + str(i) for i in range(x.shape[1])]

        predictions = []
        for tree in self.trees:
            predictions.append(
                BaggedForestPredictor.prediction_of_tree(tree, labels_description, x)
            )

        ones = csr_matrix(predictions).count_nonzero()
        zeros = len(predictions) - ones
        if ones > zeros:
            return 1
        else:
            return 0

    @staticmethod
    def prediction_of_tree(tree, labels, x):
        label = list(tree.keys())[0]
        node = tree[label]
        feature_index = labels.index(label)

        for key in node.keys():
            if x[0, feature_index] == key:
                if type(node[key]).__name__ == 'dict':
                    classified_label = BaggedForestPredictor.prediction_of_tree(node[key], labels, x)
                else:
                    classified_label = node[key]

        return classified_label
