from data_set_loader import DataSetLoader
import itertools
import numpy as np
from random import randrange
from matplotlib import pyplot as plt


class CrossValidatorTester:
    @staticmethod
    def plot(error_rates, title):
        plt.title(title)
        plt.plot(error_rates)
        plt.xlabel('Epoch')
        plt.ylabel('Total Loss')
        plt.show()

    @staticmethod
    def get_data_for(files):
        features = []
        labels = []

        for data_set_filename in files:
            new_features, new_labels = DataSetLoader(data_set_filename).load()
            features += new_features
            labels += new_labels

        return features, labels

    def __init__(self, cls, hyper_parameters, training_files, development_file, test_file):
        self.cls = cls
        self.hyper_parameters_names = list(hyper_parameters.keys())
        self.hyper_parameters = [list(x) for x in itertools.product(*hyper_parameters.values())]
        self.training_files = training_files
        self.development_file = development_file
        self.test_file = test_file

    def run(self):
        best_hyperparameters, error_rate = self.detect_best_hyperparameters()
        print(
            'BEST HYPER-PARAMETERS: %s CROSS VALIDATION ACCURACY %.2f%%' % (
                self.get_print_value(best_hyperparameters),
                100 - error_rate
            )
        )

        w = self.train(best_hyperparameters)
        test_features, test_labels = DataSetLoader(self.test_file).load()

        test_error_rate = self.calculate_error_rate(test_features, test_labels, w)
        print('\nTesting data error rate: %.2f%% TEST SET ACCURACY %.2f%%' % (test_error_rate, 100 - test_error_rate))

    def detect_best_hyperparameters(self):
        lowest_error_rate = float("inf")
        best_hyperparameters = None

        for hyper_parameter_list in self.hyper_parameters:
            perceptron_cls_init_parameters = hyper_parameter_list[:]

            error_rates = self.get_cross_validation_error_rates_for(perceptron_cls_init_parameters)
            average_error_rate = round(sum(error_rates) / len(error_rates), 2)

            print(
                'Perceptron type: %s    %s Epochs: %s    Error rates: %s    Average error rate: %.2f%%' %
                (
                    self.cls.__name__,
                    self.get_print_value(hyper_parameter_list),
                    '10',
                    "% ".join(format(e, "7.2f") for e in error_rates),
                    average_error_rate
                )
            )

            if average_error_rate < lowest_error_rate:
                lowest_error_rate = average_error_rate
                best_hyperparameters = hyper_parameter_list

        return best_hyperparameters, lowest_error_rate

    def get_cross_validation_error_rates_for(self, perceptron_cls_init_parameters):
        error_rates = []

        for idx, _ in enumerate(self.training_files):
            folds = self.training_files[:]
            test_fold_filename = folds[idx]
            del (folds[idx])

            features, labels = CrossValidatorTester.get_data_for(folds)
            test_features, test_labels = DataSetLoader(test_fold_filename).load()

            perceptron = self.cls(*perceptron_cls_init_parameters)
            weights = perceptron.train(features, labels, 10)

            invalid_entries = 0

            for i, x in enumerate(test_features):
                y1 = self.cls.predict(x, weights)
                y = test_labels[i]

                if y1 != y:
                    invalid_entries += 1

            error_rates.append(round((invalid_entries / len(test_features)) * 100, 2))

        return error_rates

    def train(self, perceptron_cls_init_parameters):
        error_rates = []
        best_error_rate = float('inf')
        best_w = None
        total_updates = 0

        perceptron = self.cls(*perceptron_cls_init_parameters)

        features, labels = CrossValidatorTester.get_data_for(self.training_files)
        development_features, development_labels = DataSetLoader(self.development_file).load()
        w = np.array([randrange(-100, 100, 1) / 10000 for _ in range(len(features[0]))])

        for _ in range(20):
            w, updates_count = perceptron.train_one_epoch(features, labels, w)
            error_rate = self.calculate_error_rate(
                development_features,
                development_labels,
                w
            )

            if error_rate < best_error_rate:
                best_error_rate = error_rate
                total_updates = updates_count
                best_w = w

            error_rates.append(error_rate)

        print('\nDevelopment set error rates: %s' % "% ".join(format(e, "7.2f") for e in error_rates))
        print(
            'Minimum error rate: %.2f%% Epoch: %d DEVELOPMENT SET ACCURACY %.2f%% UPDATES PERFORMED DURING TRAINING %d'
            % (
                min(error_rates),
                error_rates.index(min(error_rates)) + 1,
                100 - min(error_rates),
                total_updates
            )
        )

        title = 'Perceptron type: {} {} '.format(
            self.cls.__name__,
            self.get_print_value(perceptron_cls_init_parameters)
        )
        CrossValidatorTester.plot(error_rates, title)
        return best_w

    def calculate_error_rate(self, test_features, test_labels, weights):
        invalid_entries = 0

        for i, x in enumerate(test_features):
            y1 = self.cls.predict(x, weights)
            y = test_labels[i]

            if y1 != y:
                invalid_entries += 1

        return (invalid_entries / len(test_features)) * 100

    def get_print_value(self, hyper_parameter_list):
        hyper_parameters_names = self.hyper_parameters_names[:]
        hyper_parameters_names.append('')

        return ': {:.2f}    '.join(hyper_parameters_names).format(*hyper_parameter_list[:])
