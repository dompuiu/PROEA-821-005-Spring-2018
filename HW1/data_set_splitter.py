class DataSetSplitter:
    def __init__(self, data_set, column, filter_value):
        self.data_set = data_set
        self.column = column
        self.filter_value = filter_value

    def new_data_set(self):
        new_data_set = []

        for feature_vector in self.data_set:
            if feature_vector[self.column] == self.filter_value:
                reduced_feat_vec = feature_vector[:self.column]
                reduced_feat_vec.extend(feature_vector[self.column + 1:])
                new_data_set.append(reduced_feat_vec)

        return new_data_set
