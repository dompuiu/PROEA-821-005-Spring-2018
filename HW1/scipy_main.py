# This is a test to test my implementation

from sklearn import tree
import graphviz
from data_set_loader import DataSetLoader
from data_set_features_enricher import DataSetFeaturesEnricher


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

features = []
labels = []
for entry in data_set:
    features.append(entry[:-1])
    labels.append(entry[-1])

test_data = DataSetLoader('dataset/test.data').load()
enricher = DataSetFeaturesEnricher(test_data, feature_creation_labels)
test_data_set = enricher.get_enrich_data_set()

test_features = []
test_labels = []

for entry in test_data_set:
    test_features.append(entry[:-1])
    test_labels.append(entry[-1])

clf = tree.DecisionTreeClassifier()
clf.fit(features, labels)

predicted = clf.predict(test_features)
counter = 0
for idx, val in enumerate(predicted):
    if predicted[idx] != test_labels[idx]:
        counter += 1

print(test_labels)
print(predicted)
print('Error rate', counter/float(len(predicted)))

dot_data = tree.export_graphviz(
    clf,
    out_file=None,
    feature_names=feature_creation_labels,
    class_names=['+', '-'],
    filled=True, rounded=True,
    special_characters=True
)

graph = graphviz.Source(dot_data)
graph.render('mytree')
