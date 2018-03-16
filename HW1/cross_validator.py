from decision_tree import DecisionTree
from data_set_features_enricher import DataSetFeaturesEnricher
from data_set_loader import DataSetLoader
from data_set_classifier import DataSetClassifier
from classifier import Classifier
from tree_pruner import TreePruner

class CrossValidator:
    depths = [1, 2, 3, 4, 5, 10]
    feature_creation_labels = [
        'first_name_longer_that_last_name',
        'has_middle_name',
        'first_name_starts_and_ends_with_same_letter',
        'first_name_come_alphabetically_before_their_last_name',
        'second_letter_of_their_first_name_a_vowel',
        'is_the_number_of_last_name_letters_even'
    ]

    @staticmethod
    def get_data_set_for(files):
        data_set = []
        for data_set_filename in files:
            data_set += DataSetLoader(data_set_filename).load()

        return data_set

    def __init__(self, files):
        self.files = files

    def run(self):
        errors_at_depth = {}

        for depth in CrossValidator.depths:
            error_rates = self.calculate_error_rates_at_depth(depth)

            errors_at_depth[depth] = round(sum(error_rates) / len(error_rates), 2)

        minimum_error_depth = min(errors_at_depth, key=errors_at_depth.get)
        print(
            'Using the k-fold cross validation the minimum error rate was found for depth',
            minimum_error_depth,
            '\nThe error rate at depth', minimum_error_depth, 'is', errors_at_depth[minimum_error_depth], '%'
        )

        self.train_and_test_final_tree(minimum_error_depth)

    def calculate_error_rates_at_depth(self, depth):
        error_rates = []

        for idx, _ in enumerate(self.files):
            folds = self.files[:]
            test_fold_filename = folds[idx]
            del (folds[idx])

            original_data_set = CrossValidator.get_data_set_for(folds)
            enricher = DataSetFeaturesEnricher(original_data_set, CrossValidator.feature_creation_labels)

            data_set = enricher.get_enrich_data_set()
            tree = DecisionTree(data_set, CrossValidator.feature_creation_labels, depth).make_tree()
            pruned_tree = TreePruner(tree).prune()

            cls = Classifier(pruned_tree, CrossValidator.feature_creation_labels)

            dsc = DataSetClassifier(cls, enricher)
            testing_data_set = DataSetLoader(test_fold_filename).load()
            dsc.classify_data_set(testing_data_set)

            error_rates.append(dsc.error_rate)

        return error_rates

    def train_and_test_final_tree(self, depth):
        original_data_set = CrossValidator.get_data_set_for(self.files)
        enricher = DataSetFeaturesEnricher(original_data_set, CrossValidator.feature_creation_labels)

        data_set = enricher.get_enrich_data_set()
        tree = DecisionTree(data_set, CrossValidator.feature_creation_labels, depth).make_tree()
        pruned_tree = TreePruner(tree).prune()

        cls = Classifier(pruned_tree, CrossValidator.feature_creation_labels)

        dsc = DataSetClassifier(cls, enricher)
        testing_data_set = DataSetLoader('dataset/test.data').load()
        dsc.classify_data_set(testing_data_set)

        print(
            'The error rate for the test data is: ', round(dsc.error_rate, 2), '%'
        )
