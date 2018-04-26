# This is a test to test my implementation
import time
from sklearn import tree
import graphviz
from data_set_loader import DataSetLoader
from label_writer import LabelWriter

feature_creation_labels = [
    'length of screen name',
    'length of description',
    'longevity: days',
    'longevity: hours',
    'longevity: minutes',
    'longevity: seconds',
    'number of following',
    'numberof followers',
    'the ratio of the number of following and followers',
    'the number of posted tweets',
    'the number of posted tweets per day',
    'the average number of links in tweets',
    'the average number of unique links in tweets',
    'the average numer of username in tweets',
    'the average numer of unique username in tweets',
    'the change rate of number of following'
]

features, labels = DataSetLoader('../TwitterDataset/data-splits/data.train').load()
test_features, test_labels = DataSetLoader('../TwitterDataset/data-splits/data.test').load()
clf = tree.DecisionTreeClassifier()
clf.fit(features, labels)

predicted = clf.predict(test_features)
counter = 0
for idx, val in enumerate(predicted):
    if predicted[idx] != test_labels[idx]:
        counter += 1

print('Error rate', counter / float(len(predicted)) * 100)

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

timestamp = time.strftime('%b-%d-%Y_%H%M', time.localtime())
LabelWriter(
    '../TwitterDataset/data-splits/data.eval.anon',
    '../TwitterDataset/data-splits/data.eval.id',
    '../TwitterDataset/output/05-decision-tree' + timestamp + '.csv',
    clf
).write()
