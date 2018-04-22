from data_set_loader import DataSetLoader
from simple_perceptron import SimplePerceptron


class SimplePerceptronTester:
    def __init__(self, learning_rates, training_file, testing_file):
        self.learning_rates = learning_rates
        self.training_file = training_file
        self.testing_file = testing_file

    def run(self, show_weights=False):
        for learning_rate in self.learning_rates:
            print('\nSimple perceptron with learning rate %.2f' % learning_rate)

            features, labels = DataSetLoader(self.training_file).load()
            perceptron = SimplePerceptron(learning_rate)
            weights = perceptron.train(features, labels, 20)

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
