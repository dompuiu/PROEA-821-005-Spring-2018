from simple_perceptron_tester import SimplePerceptronTester

print('Simple perceptron with learning rate 1')
SimplePerceptronTester(1, 'dataset/phishing.train', 'dataset/phishing.test').run()

print('\nSimple perceptron with learning rate 0.1')
SimplePerceptronTester(0.1, 'dataset/phishing.train', 'dataset/phishing.test').run()

print('\nSimple perceptron with learning rate 0.01')
SimplePerceptronTester(0.01, 'dataset/phishing.train', 'dataset/phishing.test').run()
