from data_set_loader import DataSetLoader
from majority_baseline_classifier import MajorityBaselineClassifier


class MajorityBaselineClassifierTester:
    def __init__(self, training_file, development_file, testing_file):
        self.training_file = training_file
        self.development_file = development_file
        self.testing_file = testing_file

    def run(self):
        train_features, train_labels = DataSetLoader(self.training_file).load()
        dev_features, dev_labels = DataSetLoader(self.development_file).load()
        test_features, test_labels = DataSetLoader(self.testing_file).load()

        train_error_rate =\
            MajorityBaselineClassifierTester.error_rate(train_features, train_labels, dev_features, dev_labels)
        print('Majority Baseline accuracy for dev set: %.2f%%\n' % round(100 - train_error_rate, 2))

        dev_error_rate = \
            MajorityBaselineClassifierTester.error_rate(train_features, train_labels, test_features, test_labels)
        print('Majority Baseline accuracy for test set: %.2f%%\n' % round(100-dev_error_rate, 2))

    @staticmethod
    def error_rate(features, labels, test_features, test_labels):
        majority_label = MajorityBaselineClassifier.train(features, labels)

        invalid_entries = 0
        for i, x in enumerate(test_features):
            y1 = MajorityBaselineClassifier.predict(x, majority_label)
            y = test_labels[i]

            if y1 != y:
                invalid_entries += 1

        return (invalid_entries / len(test_features)) * 100
