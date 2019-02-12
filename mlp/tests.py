import unittest
import numpy as np
from mlp.neuralnetwork import NN
from mlp.utils import load_mnist_data
import time


class Test(unittest.TestCase):

    def test_validate_gradient(self):
        train_data, _, _ = load_mnist_data()
        N = [k*10**i for i in range(0, 5) for k in [1, 5]]
        perceptron = NN()
        perceptron.load_state('best_checkpoint',
            'epochs=10,hidden_dims=[700,500],mu=0.3112553202446868,batch_size=32,weight_init=glorot.pkl',
            zipped=True)
        perceptron.batch_size = 1
        perceptron.validate_gradient = True
        perceptron.debug = False
        with open('reports/gradient_{}.csv'.format(perceptron.params_str()), 'w') as report:
            report.write("n,epsilon,max_deviation\n")
            for n in reversed(N):
                epsilon = 1 / n
                perceptron.epsilon = epsilon
                max_deviation = perceptron.train(train_data=train_data, nb_classes=10)
                report.write("{},{},{}\n".format(n, epsilon, max_deviation))

    def test_mnist(self):
        train_data, validation_data, test_data = load_mnist_data()

        perceptron = NN(epochs=1, hidden_dims=[10, 20], mu=0.1, batch_size=50)
        perceptron.test_data = test_data
        perceptron.validation_data = validation_data

        start = time.time()
        perceptron.train(train_data=train_data, nb_classes=10)
        stop = time.time()
        print("Time: %s" % (stop - start))
        #perceptron.save_state("datasets/mnist.pkl")

        prediction = perceptron.test(test_data[:, :-1])  # pass only the features without labels
        expected = test_data[:, -1].astype(int)  # labels
        print("\nError rate: ", 1 - np.mean(prediction == expected))
