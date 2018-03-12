from entropy import Entropy
from data_set_splitter import DataSetSplitter


class BestFeatureFinder:
    def __init__(self, data_set):
        self.data_set = data_set
        self.data_set_entries_count = len(data_set)
        self.number_of_features = len(data_set[0]) - 1
        self.base_entropy = Entropy(data_set).value()

    def __entropy_for_feature(self, feature_number):
        feature_list = [example[feature_number] for example in self.data_set]
        unique_feature_values = set(feature_list)

        entropy = 0.0

        for value in unique_feature_values:
            sub_data_set = DataSetSplitter(self.data_set, feature_number, value).new_data_set()

            probability = len(sub_data_set) / float(self.data_set_entries_count)
            entropy += probability * Entropy(sub_data_set).value()

        return entropy

    def best_feature(self):
        best_info_gain = 0.0
        best_feature = -1

        for i in range(self.number_of_features):
            new_entropy = self.__entropy_for_feature(i)
            info_gain = self.base_entropy - new_entropy

            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = i

        return best_feature
