from best_feature_finder import BestFeatureFinder
from data_set_splitter import DataSetSplitter


class DecisionTree:
    def __init__(self, data_set, labels, max_depth=None):
        self.data_set = data_set
        self.labels = labels
        self.max_depth = max_depth

    @staticmethod
    def majority_count(labels_list):
        labels_count = {}
        for vote in labels_list:
            labels_count[vote] = labels_count.get(vote, 0) + 1

        sorted_labels_list = [(k, labels_count[k]) for k in sorted(labels_count, key=labels_count.get, reverse=True)]
        return sorted_labels_list[0][0]

    def make_tree(self):
        return self.__make_tree(self.data_set, self.labels, self.max_depth)

    def __make_tree(self, data_set, labels, max_depth):
        labels_list = [example[-1] for example in data_set]

        # If all examples are have same label return that label
        if labels_list.count(labels_list[0]) == len(labels_list):
            return labels_list[0]

        # When no more features or the maximum tree depth was fullfilled, return majority.
        if len(data_set[0]) == 1 or max_depth == 0:
            return DecisionTree.majority_count(labels_list)

        best_feature = BestFeatureFinder(data_set).best_feature()
        best_feature_label = labels[best_feature]

        tree = {
            best_feature_label: {}
        }

        feature_values = [example[best_feature] for example in data_set]
        unique_feature_values = set(feature_values)

        for value in unique_feature_values:
            sub_labels = [label for i, label in enumerate(labels) if i != best_feature]

            new_max_depth = max_depth
            if new_max_depth and new_max_depth > 0:
                new_max_depth -= 1

            tree[best_feature_label][value] = self.__make_tree(
                DataSetSplitter(data_set, best_feature, value).new_data_set(),
                sub_labels,
                new_max_depth
            )

        return tree
