from data_set_loader import DataSetLoader
from bagged_forest import BaggedForest, BaggedForestPredictor
from scipy.sparse import csr_matrix, vstack
import itertools
import math
import numpy as np
from inspect import signature
import pickle


class EnsembleValidatorTester:
    @staticmethod
    def split_data(all_features, all_labels, cross_validation_epoch):
        rows = all_labels.shape[1]
        split_size = math.ceil(rows / 5)
        features_splits = [all_features[i:i + split_size] for i in range(0, rows, split_size)]
        labels_splits = [all_labels.T[i:i + split_size] for i in range(0, rows, split_size)]

        features = vstack([x for i, x in enumerate(features_splits) if i != cross_validation_epoch])
        labels = vstack([x for i, x in enumerate(labels_splits) if i != cross_validation_epoch])

        test_features = features_splits[cross_validation_epoch]
        test_labels = labels_splits[cross_validation_epoch]

        return features, labels.T, test_features, test_labels.T

    def __init__(self, cls, predictor_cls, hyper_parameters, hyper_parameter_epochs, training_file, test_file, zeros, train_epochs, trees_count, features_count=67692):
        self.cls = cls
        self.predictor_cls = predictor_cls
        self.hyper_parameters_names = list(hyper_parameters.keys())
        self.hyper_parameter_epochs = hyper_parameter_epochs
        self.hyper_parameters = [list(x) for x in itertools.product(*hyper_parameters.values())]
        self.training_file = training_file
        self.test_file = test_file
        self.zeros = zeros
        self.train_epochs = train_epochs
        self.trees_count = trees_count
        self.features_count = features_count

    def run(self):
        train_features, train_labels = DataSetLoader(self.training_file, self.features_count).load(True)
        test_features, test_labels = DataSetLoader(self.test_file, self.features_count).load(self.zeros)

        file = True
        try:
            b = open('mytrees.bin', 'rb')
        except FileNotFoundError:
            file = False

        if not file:
            classifier = BaggedForest(self.trees_count)
            trees = classifier.train(train_features, train_labels)
            binary_file = open('mytrees.bin', mode='wb')
            pickle.dump(trees, binary_file)
            binary_file.close()
        else:
            print('Skipping trees generation. Found previous trees in file.')
            trees = pickle.load(b)

        features_list = self.generate_features_list(train_features, trees)
        best_hyperparameters, error_rate = self.detect_best_hyperparameters(features_list, train_labels)
        print(
            'BEST HYPER-PARAMETERS: %s CROSS VALIDATION ACCURACY: %.2f%%' % (
                self.get_print_value(best_hyperparameters),
                100 - error_rate
            )
        )

        train_features, train_labels = DataSetLoader(self.training_file, self.features_count).load(self.zeros)
        w = self.train(best_hyperparameters, train_features, train_labels)

        error_rate = self.calculate_error_rate(train_features, train_labels, w)
        print('\nTraining set error rates: %.2f%%. TRAINING SET ACCURACY %.2f%%' % (error_rate, 100 - error_rate))

        error_rate = self.calculate_error_rate(test_features, test_labels, w)
        print('\nTesting set error rates: %.2f%%. TESTING SET ACCURACY %.2f%%' % (error_rate, 100 - error_rate))

    def detect_best_hyperparameters(self, features, labels):
        lowest_error_rate = float("inf")
        best_hyperparameters = None

        for hyper_parameter_list in self.hyper_parameters:
            classifier_cls_init_parameters = hyper_parameter_list[:]

            error_rates = self.get_cross_validation_error_rates_for(classifier_cls_init_parameters, features, labels)
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

    def get_cross_validation_error_rates_for(self, classifier_cls_init_parameters, all_features, all_labels):
        error_rates = []

        if not self.zeros:
            all_labels = all_labels.toarray()
            all_labels[all_labels == 0] = -1
            all_labels = csr_matrix(all_labels)

            all_features = all_features.toarray()
            all_features[all_features == 0] = -1
            all_features = csr_matrix(all_features)
        else:
            all_labels = all_labels.toarray()
            all_labels[all_labels == -1] = 0
            all_labels = csr_matrix(all_labels)

            all_features = all_features.toarray()
            all_features[all_features == -1] = 0
            all_features = csr_matrix(all_features)

        for cross_validation_epoch in range(5):
            features, labels, test_features, test_labels = \
                EnsembleValidatorTester.split_data(all_features, all_labels, cross_validation_epoch)

            classifier = self.cls(*classifier_cls_init_parameters)
            weights = classifier.train(features, labels, self.hyper_parameter_epochs)

            error_rates.append(self.calculate_error_rate(test_features, test_labels, weights))

        return error_rates

    def train(self, classifier_cls_init_parameters, features, labels):
        error_rates = []
        best_error_rate = float('inf')
        best_w = None

        classifier = self.cls(*classifier_cls_init_parameters)

        initial_w = [0 for _ in range(features[0].shape[1])]
        w = csr_matrix(initial_w, dtype=np.float128)

        train_method_parameters = EnsembleValidatorTester.get_train_method_parameters(
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

    def generate_features_list(self, entries, trees):
        file = True
        try:
            b = open('myfeatures.bin', 'rb')
        except FileNotFoundError:
            file = False

        if not file:
            predictor_cls = EnsembleFeatureCreator(trees)
            features = []

            for i, x in enumerate(entries):
                predictions = np.array(predictor_cls.predict(x))

                if not self.zeros:
                    predictions[predictions == 0] = -1

                features.append(predictions.tolist())

            features = csr_matrix(features)
            binary_file = open('myfeatures.bin', mode='wb')
            pickle.dump(features, binary_file)
            binary_file.close()
        else:
            print('Skipping features generation. Found previous features in file.')
            features = pickle.load(b)

        return features

    def get_print_value(self, hyper_parameter_list):
        hyper_parameters_names = self.hyper_parameters_names[:]
        hyper_parameters_names.append('')

        return ': {:.4f}    '.join(hyper_parameters_names).format(*hyper_parameter_list[:])

    def calculate_error_rate(self, test_features, test_labels, weights):
        invalid_entries = 0
        predictor_cls = self.predictor_cls(weights)

        for i, x in enumerate(test_features):
            y1 = predictor_cls.predict(x)
            y = test_labels.toarray()[0][i]

            if y1 != y:
                invalid_entries += 1

        return (invalid_entries / test_features.shape[0]) * 100

    @staticmethod
    def get_train_method_parameters(method, all_parameters):
        train_one_epoch_signature = signature(method)
        result = []
        for param_name in train_one_epoch_signature.parameters:
            result.append(all_parameters[param_name])

        return result


class EnsembleFeatureCreator:
    def __init__(self, trees):
        self.trees = trees

    def predict(self, x):
        labels_description = ['Feature ' + str(i) for i in range(x.shape[1])]

        predictions = []
        for tree in self.trees:
            predictions.append(
                EnsembleFeatureCreator.prediction_of_tree(tree, labels_description, x)
            )
        return predictions

    @staticmethod
    def prediction_of_tree(tree, labels, x):
        label = list(tree.keys())[0]
        node = tree[label]
        feature_index = labels.index(label)

        for key in node.keys():
            if x[0, feature_index] == key:
                if type(node[key]).__name__ == 'dict':
                    classified_label = BaggedForestPredictor.prediction_of_tree(node[key], labels, x)
                else:
                    classified_label = node[key]

        return classified_label
