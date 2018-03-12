class Classifier:
    def __init__(self, tree, labels):
        self.tree = tree
        self.labels = labels

    def classify(self, test_vector):
        label = list(self.tree.keys())[0]
        node = self.tree[label]
        feature_index = self.labels.index(label)

        for key in node.keys():
            if test_vector[feature_index] == key:
                if type(node[key]).__name__ == 'dict':
                    inner_classifier = Classifier(node[key], self.labels)
                    classified_label = inner_classifier.classify(test_vector)
                else:
                    classified_label = node[key]

        return classified_label
