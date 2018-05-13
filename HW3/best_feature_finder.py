from entropy import Entropy
from data_set_splitter import DataSetSplitter


class BestFeatureFinder:
    def __init__(self, features_list, labels_list):
        self.features_list = features_list
        self.labels_list = labels_list
        self.data_set_entries_count = features_list.shape[0]
        self.number_of_features = features_list.shape[1] - 1
        self.base_entropy = Entropy(features_list, labels_list).value()

    def entropy_for_feature(self, feature_number):
        unique_feature_values = [0, 1]

        entropy = 0.0

        for value in unique_feature_values:
            sub_features_list, sub_labels_list = \
                DataSetSplitter(self.features_list, self.labels_list, feature_number, value).new_data_set()

            probability = sub_features_list.size / float(self.data_set_entries_count)
            entropy += probability * Entropy(sub_features_list, sub_labels_list).value()

        return entropy

    def best_feature(self):
        best_info_gain = -1
        best_feature = -1

        for i in range(self.number_of_features):
            new_entropy = self.entropy_for_feature(i)
            info_gain = self.base_entropy - new_entropy

            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = i

        return best_feature
