from best_feature_finder import BestFeatureFinder
from data_set_splitter import DataSetSplitter


class DecisionTree:
    def __init__(self, data_set, labels):
        self.data_set = data_set
        self.labels = labels

    def __majority_count(self, labels_list):
        labels_count = {}
        for vote in labels_list:
            labels_count[vote] = labels_count.get(vote, 0) + 1

        sorted_labels_list = [(k, labels_count[k]) for k in sorted(labels_count, key=labels_count.get, reverse=True)]
        return sorted_labels_list[0][0]

    def make_tree(self):
        labels_list = [example[-1] for example in self.data_set]

        # If all examples are have same label return that label
        if labels_list.count(labels_list[0]) == len(labels_list):
            return labels_list[0]

        # When no more features, return majority.
        if len(self.data_set[0]) == 1:
            return self.__majority_count(labels_list)

        best_feature = BestFeatureFinder(self.data_set).best_feature()
        best_feature_label = self.labels[best_feature]

        tree = {
            best_feature_label: {}
        }

        feature_values = [example[best_feature] for example in self.data_set]
        unique_feature_values = set(feature_values)

        for value in unique_feature_values:
            sub_labels = [label for i, label in enumerate(self.labels) if i != best_feature]
            tree[best_feature_label][value] = DecisionTree(
                DataSetSplitter(self.data_set, best_feature, value).new_data_set(),
                sub_labels
            ).make_tree()

        return tree
