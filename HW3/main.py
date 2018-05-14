from cross_validator_tester import CrossValidatorTester
from svm import SVM, SVMPredictor
from logistic_regression import LogisticRegression, LogisticRegressionPredictor
from naive_bayes import NaiveBayes, NaiveBayesPredictor
from bagged_forest_validator_tester import BaggedForestValidatorTester
from ensemble_validator_tester import EnsembleValidatorTester


CrossValidatorTester(
    SVM,
    SVMPredictor,
    {
        'Initial learning rate': [10, 1, 10 ** -1, 10 ** -2, 10 ** -3],
        'Regularization loss tradeoff': [10, 1, 10 ** -1, 10 ** -2, 10 ** -3]
    },
    [
        'dataset/CVSplits/training00.data',
        'dataset/CVSplits/training01.data',
        'dataset/CVSplits/training02.data',
        'dataset/CVSplits/training03.data',
        'dataset/CVSplits/training04.data'
    ],
    'dataset/speeches.test.liblinear',
    1,
    5
).run()

CrossValidatorTester(
    LogisticRegression,
    LogisticRegressionPredictor,
    {
        'Initial learning rate': [10, 1, 10 ** -1, 10 ** -2, 10 ** -3],
        'Regularization loss tradeoff': [10, 1, 10 ** -1, 10 ** -2, 10 ** -3]
    },
    [
        'dataset/CVSplits/training00.data',
        'dataset/CVSplits/training01.data',
        'dataset/CVSplits/training02.data',
        'dataset/CVSplits/training03.data',
        'dataset/CVSplits/training04.data'
    ],
    'dataset/speeches.test.liblinear',
    30,
    30,
    True
).run()

CrossValidatorTester(
    NaiveBayes,
    NaiveBayesPredictor,
    {
        'Smoothing term': [2, 1.5, 1, 0.5]
    },
    [
        'dataset/CVSplits/training00.data',
        'dataset/CVSplits/training01.data',
        'dataset/CVSplits/training02.data',
        'dataset/CVSplits/training03.data',
        'dataset/CVSplits/training04.data'
    ],
    'dataset/speeches.test.liblinear',
    1,
    10,
    True, # If -1 should be saved as 0. SciPy sparsed matrixes has helper methods that work only for 0s and non 0s.
    200 # How many features to load from the file (don't load all features due to computation limitations)
).run()

BaggedForestValidatorTester(
    'dataset/speeches.train.liblinear',
    'dataset/speeches.test.liblinear',
    100,  # How many trees to generate
    200  # How many features to load from the file (don't load all features due to computation limitations)
).run()

EnsembleValidatorTester(
    SVM,
    SVMPredictor,
    {
        'Initial learning rate': [1, 10 ** -1, 10 ** -2, 10 ** -3, 10 ** -4],
        'Regularization loss tradeoff': [10, 1, 10 ** -1, 10 ** -2, 10 ** -3, 10 ** -4]
    },
    1,  # How many epochs to train while determining best hyper parameters
    'dataset/speeches.train.liblinear',
    'dataset/speeches.test.liblinear',
    False,
    5,  # How many epochs when training the classifier
    200,  # How many trees to generate for feature transformation
    200  # How many features to load from the file (don't load all features due to computation limitations)
).run()

EnsembleValidatorTester(
    LogisticRegression,
    LogisticRegressionPredictor,
    {
        'Initial learning rate': [1, 10 ** -1, 10 ** -2, 10 ** -3, 10 ** -4],
        'Regularization loss tradeoff': [10, 1, 10 ** -1, 10 ** -2, 10 ** -3, 10 ** -4]
    },
    100,  # How many epochs to train while determining best hyper parameters
    'dataset/speeches.train.liblinear',
    'dataset/speeches.test.liblinear',
    True,
    100,  # How many epochs when training the classifier
    200,  # How many trees to generate for feature transformation
    200  # How many features to load from the file (don't load all features due to computation limitations)
).run()
