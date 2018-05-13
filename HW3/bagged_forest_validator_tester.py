from data_set_loader import DataSetLoader
from bagged_forest import BaggedForest, BaggedForestPredictor


class BaggedForestValidatorTester:
    def __init__(self, training_file, test_file, trees_count, features_count=67692):
        self.training_file = training_file
        self.test_file = test_file
        self.trees_count = trees_count
        self.features_count = features_count

    def run(self):
        train_features, train_labels = DataSetLoader(self.training_file, self.features_count).load(True)
        test_features, test_labels = DataSetLoader(self.test_file, self.features_count).load(True)

        classifier = BaggedForest(self.trees_count)
        trees = classifier.train(train_features, train_labels)

        error_rate = BaggedForestValidatorTester.calculate_error_rate(train_features, train_labels, trees)
        print('\nTraining set error rates: %.2f%%. TRAINING SET ACCURACY %.2f%%' % (error_rate, 100 - error_rate))

        error_rate = BaggedForestValidatorTester.calculate_error_rate(test_features, test_labels, trees)
        print('\nTesting set error rates: %.2f%%. TESTING SET ACCURACY %.2f%%' % (error_rate, 100 - error_rate))

    @staticmethod
    def calculate_error_rate(test_features, test_labels, trees):
        test_labels = test_labels
        invalid_entries = 0

        predictor_cls = BaggedForestPredictor(trees)

        for i, x in enumerate(test_features):
            y1 = predictor_cls.predict(x)
            y = test_labels.toarray()[0][i]

            if y1 != y:
                invalid_entries += 1

        return (invalid_entries / test_features.shape[0]) * 100

