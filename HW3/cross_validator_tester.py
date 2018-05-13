from data_set_loader import DataSetLoader
import itertools
import numpy as np
from inspect import signature
from scipy.sparse import csr_matrix, vstack, hstack


class CrossValidatorTester:
    @staticmethod
    def get_data_for(files, zeros, features_count):
        first_time = True
        features = None
        labels = None

        for data_set_filename in files:
            new_features, new_labels = DataSetLoader(data_set_filename, features_count).load(zeros)
            if first_time:
                features = new_features
                labels = new_labels
                first_time = False
            else:
                features = vstack([features, new_features])
                labels = hstack([labels, new_labels])

            # labels += new_labels

        return features, labels

    def __init__(self, cls, predictor_cls, hyper_parameters, training_files, test_file, hyper_parameter_epochs=1, train_epochs=1, zeros=False, features=67692):
        self.cls = cls
        self.predictor_cls = predictor_cls
        self.hyper_parameters_names = list(hyper_parameters.keys())
        self.hyper_parameters = [list(x) for x in itertools.product(*hyper_parameters.values())]
        self.training_files = training_files
        self.test_file = test_file
        self.hyper_parameter_epochs = hyper_parameter_epochs
        self.train_epochs = train_epochs
        self.zeros = zeros
        self.features = features

    def run(self):
        best_hyperparameters, error_rate = self.detect_best_hyperparameters()
        print(
            'BEST HYPER-PARAMETERS: %s CROSS VALIDATION ACCURACY: %.2f%%' % (
                self.get_print_value(best_hyperparameters),
                100 - error_rate
            )
        )

        w = self.train(best_hyperparameters)
        test_features, test_labels = DataSetLoader(self.test_file, self.features).load(self.zeros)

        test_error_rate = self.calculate_error_rate(test_features, test_labels, w)
        print('\nTesting data error rate: %.2f%% TEST SET ACCURACY %.2f%%' % (test_error_rate, 100 - test_error_rate))

    def detect_best_hyperparameters(self):
        lowest_error_rate = float("inf")
        best_hyperparameters = None

        for hyper_parameter_list in self.hyper_parameters:
            classifier_cls_init_parameters = hyper_parameter_list[:]

            error_rates = self.get_cross_validation_error_rates_for(classifier_cls_init_parameters)
            average_error_rate = round(sum(error_rates) / len(error_rates), 2)

            print(
                'Classifier type: %s    %s Epochs: %s    Error rates: %s    Average error rate: %.2f%%' %
                (
                    self.cls.__name__,
                    self.get_print_value(hyper_parameter_list),
                    self.hyper_parameter_epochs,
                    "% ".join(format(e, "7.2f") for e in error_rates),
                    average_error_rate
                )
            )

            if average_error_rate < lowest_error_rate:
                lowest_error_rate = average_error_rate
                best_hyperparameters = hyper_parameter_list

        return best_hyperparameters, lowest_error_rate

    def get_cross_validation_error_rates_for(self, classifier_cls_init_parameters):
        error_rates = []

        for idx, _ in enumerate(self.training_files):
            folds = self.training_files[:]
            test_fold_filename = folds[idx]
            del (folds[idx])

            features, labels = CrossValidatorTester.get_data_for(folds, self.zeros, self.features)
            test_features, test_labels = DataSetLoader(test_fold_filename, self.features).load(self.zeros)

            classifier = self.cls(*classifier_cls_init_parameters)
            weights = classifier.train(features, labels, self.hyper_parameter_epochs)

            error_rates.append(self.calculate_error_rate(test_features, test_labels, weights))

        return error_rates

    def train(self, classifier_cls_init_parameters):
        error_rates = []
        best_error_rate = float('inf')
        best_w = None

        classifier = self.cls(*classifier_cls_init_parameters)

        features, labels = CrossValidatorTester.get_data_for(self.training_files, self.zeros, self.features)

        initial_w = [0 for _ in range(features[0].shape[1])]
        w = csr_matrix(initial_w, dtype=np.float128)

        train_method_parameters = self.get_train_method_parameters(
            classifier.train_one_epoch,
            {
                'train': features,
                'labels': labels,
                'w': w,
                'c': 1,
                'epoch': 0
            }
        )

        for _ in range(self.train_epochs):
            new_train_method_parameters = classifier.train_one_epoch(*train_method_parameters)
            new_train_method_parameters.insert(0, labels)
            new_train_method_parameters.insert(0, features)

            train_method_parameters = new_train_method_parameters
            w = train_method_parameters[2]

            error_rate = self.calculate_error_rate(
                features,
                labels,
                w
            )

            if error_rate < best_error_rate:
                best_error_rate = error_rate
                best_w = w

            error_rates.append(error_rate)

        print('\nTraining set error rates: %s' % "% ".join(format(e, "7.2f") for e in error_rates))
        print(
            'Minimum error rate: %.2f%% Epoch: %d TRAINING SET ACCURACY %.2f%%'
            % (
                min(error_rates),
                error_rates.index(min(error_rates)),
                100 - min(error_rates)
            )
        )

        return best_w

    def calculate_error_rate(self, test_features, test_labels, weights):
        test_labels = test_labels
        invalid_entries = 0
        predictor_cls = self.predictor_cls(weights)

        for i, x in enumerate(test_features):
            y1 = predictor_cls.predict(x)
            y = test_labels.toarray()[0][i]

            if y1 != y:
                invalid_entries += 1

        return (invalid_entries / test_features.shape[0]) * 100

    def get_print_value(self, hyper_parameter_list):
        hyper_parameters_names = self.hyper_parameters_names[:]
        hyper_parameters_names.append('')

        return ': {:.3f}    '.join(hyper_parameters_names).format(*hyper_parameter_list[:])

    def get_train_method_parameters(self, method, all_parameters):
        train_one_epoch_signature = signature(method)
        result = []
        for param_name in train_one_epoch_signature.parameters:
            result.append(all_parameters[param_name])

        return result
