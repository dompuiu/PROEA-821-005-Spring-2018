from simple_perceptron_tester import SimplePerceptronTester
from dynamic_learning_rate_perceptron_tester import DynamicLearningRatePerceptronTester
from margin_perceptron_tester import MarginPerceptronTester
from averaged_perceptron_tester import AveragedPerceptronTester

# Simple perceptron
print('\n######################################################')
print('\nSimple perceptron - Testing perceptron on training set')
print('\n######################################################')
SimplePerceptronTester([1, 0.1, 0.01], 'dataset/phishing.train', 'dataset/phishing.train').run()

print('\n#####################################################')
print('\nSimple perceptron - Testing perceptron on testing set')
print('\n#####################################################')
SimplePerceptronTester([1, 0.1, 0.01], 'dataset/phishing.train', 'dataset/phishing.test').run()

# Dynamic learning rate
print('\n#####################################################################')
print('\nDynamic learning rate perceptron - Testing perceptron on training set')
print('\n#####################################################################')
DynamicLearningRatePerceptronTester([1, 0.1, 0.01], 'dataset/phishing.train', 'dataset/phishing.train').run()

print('\n####################################################################')
print('\nDynamic learning rate perceptron - Testing perceptron on testing set')
print('\n####################################################################')
DynamicLearningRatePerceptronTester([1, 0.1, 0.01], 'dataset/phishing.train', 'dataset/phishing.test').run()


# Margin perceptron tester
print('\n######################################################')
print('\nMargin perceptron - Testing perceptron on training set')
print('\n######################################################')
MarginPerceptronTester([1, 0.1, 0.01], [1, 0.1, 0.01], 'dataset/phishing.train', 'dataset/phishing.train').run()

print('\n#####################################################')
print('\nMargin perceptron - Testing perceptron on testing set')
print('\n#####################################################')
MarginPerceptronTester([1, 0.1, 0.01], [1, 0.1, 0.01], 'dataset/phishing.train', 'dataset/phishing.test').run()

# Averaged perceptron
print('\n########################################################')
print('\nAveraged perceptron - Testing perceptron on training set')
print('\n########################################################')
AveragedPerceptronTester([1, 0.1, 0.01], 'dataset/phishing.train', 'dataset/phishing.train').run()

print('\n#######################################################')
print('\nAveraged perceptron - Testing perceptron on testing set')
print('\n#######################################################')
AveragedPerceptronTester([1, 0.1, 0.01], 'dataset/phishing.train', 'dataset/phishing.test').run()
