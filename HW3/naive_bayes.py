import numpy as np


class NaiveBayes:
    def __init__(self, smoothing=1):
        self.smoothing = smoothing

    def train(self, train, labels, _):
        [w] = self.train_one_epoch(train, labels)
        return w

    def train_one_epoch(self, train, labels, w=None):
        w = {}
        smoothing = self.smoothing

        total_labels = labels.shape[1]

        # Calculate priors
        prior_y_1 = labels.count_nonzero()
        prior_y_0 = total_labels - prior_y_1

        w[('prior', 0)] = prior_y_0 / total_labels
        w[('prior', 1)] = prior_y_1 / total_labels

        zero_indices = np.where(labels.toarray() == 0)[1]
        one_indices = labels.nonzero()[1]

        for feature_indx in range(train.shape[1]):
            # if feature_indx % 10000 == 0:
            #     print(feature_indx)

            feature_values = train.getcol(feature_indx)

            feature_values_given_label_1 = feature_values[one_indices]
            count_feature_total_values_given_label_1 = feature_values_given_label_1.shape[0]

            count_feature_has_value_1_given_label_1 = feature_values_given_label_1.count_nonzero()
            count_feature_has_value_0_given_label_1 = \
                count_feature_total_values_given_label_1 - count_feature_has_value_1_given_label_1

            w[(feature_indx, 1, 1)] = (count_feature_has_value_1_given_label_1 + smoothing) / \
                                      (count_feature_total_values_given_label_1 + smoothing * 2)

            w[(feature_indx, 0, 1)] = (count_feature_has_value_0_given_label_1 + smoothing) / \
                                      (count_feature_total_values_given_label_1 + smoothing * 2)

            feature_values_given_label_0 = feature_values[zero_indices]
            count_feature_total_values_given_label_0 = feature_values_given_label_0.shape[0]

            count_feature_has_value_1_given_label_0 = feature_values_given_label_0.count_nonzero()
            count_feature_has_value_0_given_label_0 = \
                count_feature_total_values_given_label_0 - count_feature_has_value_1_given_label_0

            w[(feature_indx, 1, 0)] = (count_feature_has_value_1_given_label_0 + smoothing) / \
                                      (count_feature_total_values_given_label_0 + smoothing * 2)
            w[(feature_indx, 0, 0)] = (count_feature_has_value_0_given_label_0 + smoothing) / \
                                      (count_feature_total_values_given_label_0 + smoothing * 2)

        return [w]


class NaiveBayesPredictor:
    def __init__(self, w):
        self.w = w

    def predict(self, x):
        y1 = self.w[('prior', 1)]
        y0 = self.w[('prior', 0)]

        for idx in range(x.shape[1]):
            x_val = x[0, idx]

            y1 *= self.w[(idx, x_val, 1)]
            y0 *= self.w[(idx, x_val, 0)]

        if y1 > y0:
            return 1
        else:
            return 0
