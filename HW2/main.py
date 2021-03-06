from simple_perceptron import SimplePerceptron
from dynamic_learning_rate_perceptron import DynamicLearningRatePerceptron
from margin_perceptron import MarginPerceptron
from averaged_perceptron import AveragedPerceptron
from cross_validator_tester import CrossValidatorTester
from majority_baseline_classifier_tester import MajorityBaselineClassifierTester

print('\n#################')
print('\nMajority Baseline')
print('\n#################')

MajorityBaselineClassifierTester(
    'dataset/phishing.train',
    'dataset/phishing.dev',
    'dataset/phishing.test'
).run()

print('\n#################')
print('\nSimple perceptron')
print('\n#################')

CrossValidatorTester(
    SimplePerceptron,
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
    'dataset/phishing.dev',
    'dataset/phishing.test'
).run()


print('\n################################')
print('\nDynamic learning rate perceptron')
print('\n################################')

CrossValidatorTester(
    DynamicLearningRatePerceptron,
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
    'dataset/phishing.dev',
    'dataset/phishing.test'
).run()

print('\n#################')
print('\nMargin perceptron')
print('\n#################')

CrossValidatorTester(
    MarginPerceptron,
    {
        'Learning rate': [1, 0.1, 0.01],
        'Margin': [1, 0.1, 0.01]
    },
    [
        'dataset/CVSplits/training00.data',
        'dataset/CVSplits/training01.data',
        'dataset/CVSplits/training02.data',
        'dataset/CVSplits/training03.data',
        'dataset/CVSplits/training04.data'
    ],
    'dataset/phishing.dev',
    'dataset/phishing.test'
).run()

print('\n###################')
print('\nAveraged perceptron')
print('\n###################')

CrossValidatorTester(
    AveragedPerceptron,
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
    'dataset/phishing.dev',
    'dataset/phishing.test'
).run()
