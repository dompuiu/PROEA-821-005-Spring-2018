from decision_tree import DecisionTree
from tree_plotter import create_plot
from data_set_loader import DataSetLoader
from data_set_features_enricher import DataSetFeaturesEnricher
from classifier import Classifier
from tree_utils import get_tree_depth
from tree_pruner import TreePruner
from data_set_classifier import DataSetClassifier

# Load training data.
original_data_set = DataSetLoader('dataset/training.data').load()

# Create a data set using the following features.
feature_creation_labels = [
    'first_name_longer_that_last_name',
    'has_middle_name',
    'first_name_starts_and_ends_with_same_letter',
    'first_name_come_alphabetically_before_their_last_name',
    'second_letter_of_their_first_name_a_vowel',
    'is_the_number_of_last_name_letter_even'
]
enricher = DataSetFeaturesEnricher(original_data_set, feature_creation_labels)
data_set = enricher.get_enrich_data_set()

# Create a set of short labels. Having long labels made the rendered tree unreadable.
short_labels = [
    'fn_longer_ls',
    'middle',
    'f&l',
    'fn_before_ln',
    'vowel',
    'ln_even'
]

# Create the decision tree and render it.
tree = DecisionTree(data_set, short_labels).make_tree()
# create_plot(tree)

# Prune the training set.
pruned_tree = TreePruner(tree).prune()
# create_plot(pruned_tree)
print('Tree depth: ', get_tree_depth(tree))

# Classify other results
c = Classifier(pruned_tree, short_labels)

print('\nClassify the training set: ')
dsc = DataSetClassifier(c, enricher)
dsc.classify_data_set(original_data_set)

print('\nClassify the test set: ')
testing_data_set = DataSetLoader('dataset/test.data').load()
dsc.classify_data_set(testing_data_set)
