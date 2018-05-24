from svm import SVM
from cross_validator_tester import CrossValidatorTester
from label_writer import LabelWriter
import time

print('\n#################')
print('\nSVM')
print('\n#################')

w = CrossValidatorTester(
    SVM,
    {
        'Initial learning rate': [10, 1, 10 ** -1, 10 ** -2, 10 ** -3],
        'Regularization loss tradeoff': [10, 1, 10 ** -1, 10 ** -2, 10 ** -3]
    },
    [
        '../TwitterDataset/data-splits/cross-validation-splits/split01.train',
        '../TwitterDataset/data-splits/cross-validation-splits/split02.train',
        '../TwitterDataset/data-splits/cross-validation-splits/split03.train',
        '../TwitterDataset/data-splits/cross-validation-splits/split04.train',
        '../TwitterDataset/data-splits/cross-validation-splits/split05.train'
    ],
    '../TwitterDataset/data-splits/data.train',
    '../TwitterDataset/data-splits/data.test'
).run()

timestamp = time.strftime('%b-%d-%Y_%H%M', time.localtime())

LabelWriter(
    '../TwitterDataset/data-splits/data.eval.anon',
    '../TwitterDataset/data-splits/data.eval.id',
    '../TwitterDataset/output/07-svm' + timestamp + '.csv',
    w
).write()
