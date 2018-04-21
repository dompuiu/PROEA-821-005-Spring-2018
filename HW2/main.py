from simple_perceptron_tester import SimplePerceptronTester

print('Simple perceptron with learning rate 1 - Tesing on training set')
SimplePerceptronTester(1, 'dataset/phishing.train', 'dataset/phishing.train').run()

print('Simple perceptron with learning rate 1 - Tesing on testing set')
SimplePerceptronTester(1, 'dataset/phishing.train', 'dataset/phishing.test').run()

print('\nSimple perceptron with learning rate 0.1 - Tesing on training set')
SimplePerceptronTester(0.1, 'dataset/phishing.train', 'dataset/phishing.train').run()

print('\nSimple perceptron with learning rate 0.1 - Tesing on testing set')
SimplePerceptronTester(0.1, 'dataset/phishing.train', 'dataset/phishing.test').run()

print('\nSimple perceptron with learning rate 0.01 - Tesing on training set')
SimplePerceptronTester(0.01, 'dataset/phishing.train', 'dataset/phishing.train').run()

print('\nSimple perceptron with learning rate 0.01 - Tesing on tesging set')
SimplePerceptronTester(0.01, 'dataset/phishing.train', 'dataset/phishing.test').run()
