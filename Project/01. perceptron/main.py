from simple_perceptron import SimplePerceptron
from dynamic_learning_rate_perceptron import DynamicLearningRatePerceptron
from margin_perceptron import MarginPerceptron
from averaged_perceptron import AveragedPerceptron
from cross_validator_tester import CrossValidatorTester
from label_writer import LabelWriter
import time

print('\n#################')
print('\nSimple perceptron')
print('\n#################')

w = CrossValidatorTester(
    SimplePerceptron,
    {
        'Learning rate': [1, 0.1, 0.01]
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
    '../TwitterDataset/output/01-simple-perceptron' + timestamp + '.csv',
    w
).write()


print('\n################################')
print('\nDynamic learning rate perceptron')
print('\n################################')

w = CrossValidatorTester(
    DynamicLearningRatePerceptron,
    {
        'Learning rate': [1, 0.1, 0.01]
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
    '../TwitterDataset/output/02-dynamic-learning-rate-perceptron' + timestamp + '.csv',
    w
).write()

print('\n#################')
print('\nMargin perceptron')
print('\n#################')

w = CrossValidatorTester(
    MarginPerceptron,
    {
        'Learning rate': [1, 0.1, 0.01],
        'Margin': [1, 0.1, 0.01]
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
    '../TwitterDataset/output/03-margin-perceptron' + timestamp + '.csv',
    w
).write()


print('\n###################')
print('\nAveraged perceptron')
print('\n###################')

w = CrossValidatorTester(
    AveragedPerceptron,
    {
        'Learning rate': [1, 0.1, 0.01]
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
    '../TwitterDataset/output/04-averaged-perceptron' + timestamp + '.csv',
    w
).write()
