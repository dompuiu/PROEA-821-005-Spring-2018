import numpy as np
from scipy.sparse import csr_matrix


class DataSetSplitter:
    def __init__(self, featurest_list, labels_list, column, filter_value):
        self.features_list = featurest_list
        self.labels_list = labels_list
        self.column = column
        self.filter_value = filter_value

    def new_data_set(self):
        f = self.features_list.getcol(self.column)

        if self.filter_value == 1:
            indices = f.nonzero()[0]
        else:
            indices = np.where(f.toarray() == 0)[1]

        sub_features_list = csr_matrix(np.delete(self.features_list[indices].toarray(), self.column, axis=1))

        return sub_features_list, self.labels_list[indices]
