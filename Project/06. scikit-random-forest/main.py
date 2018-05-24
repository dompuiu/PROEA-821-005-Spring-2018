# This is a test to test my implementation
import time
from sklearn import ensemble
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
clf = ensemble.RandomForestClassifier()
clf.fit(features, labels)

predicted = clf.predict(features)
counter = 0
for idx, val in enumerate(predicted):
    if predicted[idx] != labels[idx]:
        counter += 1

print('Training Error rate', counter / float(len(predicted)) * 100)

predicted = clf.predict(test_features)
counter = 0
for idx, val in enumerate(predicted):
    if predicted[idx] != test_labels[idx]:
        counter += 1

print('Test Error rate', counter / float(len(predicted)) * 100)

timestamp = time.strftime('%b-%d-%Y_%H%M', time.localtime())
LabelWriter(
    '../TwitterDataset/data-splits/data.eval.anon',
    '../TwitterDataset/data-splits/data.eval.id',
    '../TwitterDataset/output/09-bagged-forest' + timestamp + '.csv',
    clf
).write()

