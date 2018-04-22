from simple_perceptron import SimplePerceptron
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

# # Simple perceptron
# print('\n######################################################')
# print('\nSimple perceptron - Testing perceptron on training set')
# print('\n######################################################')
# SimplePerceptronTester([1, 0.1, 0.01], 'dataset/phishing.train', 'dataset/phishing.train').run()
#
# print('\n#####################################################')
# print('\nSimple perceptron - Testing perceptron on testing set')
# print('\n#####################################################')
# SimplePerceptronTester([1, 0.1, 0.01], 'dataset/phishing.train', 'dataset/phishing.test').run()
#
# # Dynamic learning rate
# print('\n#####################################################################')
# print('\nDynamic learning rate perceptron - Testing perceptron on training set')
# print('\n#####################################################################')
# DynamicLearningRatePerceptronTester([1, 0.1, 0.01], 'dataset/phishing.train', 'dataset/phishing.train').run()
#
# print('\n####################################################################')
# print('\nDynamic learning rate perceptron - Testing perceptron on testing set')
# print('\n####################################################################')
# DynamicLearningRatePerceptronTester([1, 0.1, 0.01], 'dataset/phishing.train', 'dataset/phishing.test').run()
#
#
# # Margin perceptron tester
# print('\n######################################################')
# print('\nMargin perceptron - Testing perceptron on training set')
# print('\n######################################################')
# MarginPerceptronTester([1, 0.1, 0.01], [1, 0.1, 0.01], 'dataset/phishing.train', 'dataset/phishing.train').run()
#
# print('\n#####################################################')
# print('\nMargin perceptron - Testing perceptron on testing set')
# print('\n#####################################################')
# MarginPerceptronTester([1, 0.1, 0.01], [1, 0.1, 0.01], 'dataset/phishing.train', 'dataset/phishing.test').run()
#
# # Averaged perceptron
# print('\n########################################################')
# print('\nAveraged perceptron - Testing perceptron on training set')
# print('\n########################################################')
# AveragedPerceptronTester([1, 0.1, 0.01], 'dataset/phishing.train', 'dataset/phishing.train').run()
#
# print('\n#######################################################')
# print('\nAveraged perceptron - Testing perceptron on testing set')
# print('\n#######################################################')
# AveragedPerceptronTester([1, 0.1, 0.01], 'dataset/phishing.train', 'dataset/phishing.test').run()
