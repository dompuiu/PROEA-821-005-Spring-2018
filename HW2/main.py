from simple_perceptron import SimplePerceptron
from dynamic_learning_rate_perceptron import DynamicLearningRatePerceptron
from cross_validator_tester import CrossValidatorTester

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
