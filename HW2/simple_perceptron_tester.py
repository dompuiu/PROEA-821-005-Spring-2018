from data_set_loader import DataSetLoader
from simple_perceptron import SimplePerceptron


class SimplePerceptronTester:
    def __init__(self, learning_rate, training_file, testing_file):
        self.learning_rate = learning_rate
        self.training_file = training_file
        self.testing_file = testing_file

    def run(self, show_weights = False):
        features, labels = DataSetLoader(self.training_file).load()

        perceptron = SimplePerceptron()

        weights = perceptron.train_weights(features, labels)
        if show_weights:
            print('\nDetected weights')
            print(weights)

        test_features, test_labels = DataSetLoader(self.testing_file).load()

        invalid_entries = 0

        for i, x in enumerate(test_features):
            y1 = SimplePerceptron.predict(x, weights)
            y = test_labels[i]

            if y1 != y:
                invalid_entries += 1

        error_rate = (invalid_entries / len(test_features)) * 100
        print('Invalid classified entries:', invalid_entries, '-> Total entries:',
              len(test_features), '-> Error:', str(round(error_rate, 2)) + '%\n')
