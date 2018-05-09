from data_set_loader import DataSetLoader
from cross_validator_tester import CrossValidatorTester
from svm import SVM

CrossValidatorTester(
    SVM,
    {
        'Learning rate': [1, 0.1, 0.01]
    },
    [
        'dataset/CVSplits/training00.data',
        'dataset/CVSplits/training01.data',
        'dataset/CVSplits/training02.data',
        'dataset/CVSplits/training03.data',
        'dataset/CVSplits/training04.data'
    ],
    'dataset/phishing.test'
).run()
