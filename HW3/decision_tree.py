from best_feature_finder import BestFeatureFinder
from data_set_splitter import DataSetSplitter


class DecisionTree:
    def __init__(self, features_list, labels_list, labels_description, max_depth=None):
        self.features_list = features_list
        self.labels_list = labels_list
        self.labels_description = labels_description
        self.max_depth = max_depth

    @staticmethod
    def majority_count(labels_list):
        ones = labels_list.count_nonzero()
        zeros = labels_list.shape[0] - ones

        if ones > zeros:
            return 1
        else:
            return 0

    def make_tree(self):
        return self.__make_tree(self.features_list, self.labels_list, self.labels_description, self.max_depth)

    def __make_tree(self, features_list, labels_list, labels_description, max_depth):
        # If all examples are have same label return that label
        if labels_list.count_nonzero() == labels_list.shape[0]:
            return 1
        elif labels_list.count_nonzero() == 0:
            return 0

        # When no more features or the maximum tree depth was reached, return majority.
        if features_list.shape[1] == 1 or max_depth == 0:
            return DecisionTree.majority_count(labels_list)

        best_feature = BestFeatureFinder(features_list, labels_list).best_feature()
        best_feature_label = labels_description[best_feature]

        tree = {
            best_feature_label: {}
        }

        for value in [0, 1]:
            sub_labels_descriptions = [label for i, label in enumerate(labels_description) if i != best_feature]

            new_max_depth = max_depth
            if new_max_depth and new_max_depth > 0:
                new_max_depth -= 1

            sub_features_list, sub_labels_list = \
                DataSetSplitter(features_list, labels_list, best_feature, value).new_data_set()

            tree[best_feature_label][value] = self.__make_tree(
                sub_features_list,
                sub_labels_list,
                sub_labels_descriptions,
                new_max_depth
            )

        return tree
