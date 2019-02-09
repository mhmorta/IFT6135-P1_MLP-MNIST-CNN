import unittest
import numpy as np
from mlp.neuralnetwork import NN
from mlp.utils import load_mnist_data
import time


class Test(unittest.TestCase):

    def test_validate_gradient(self):
        train_data, _, _ = load_mnist_data()
        I = range(0, 5)
        K = [1, 5]
        N = [k*10**i for i in I for k in K]
        for n in reversed(N):
            epsilon = 1 / n
            perceptron = NN(epochs=1, hidden_dims=[500, 600], mu=0.1, batch_size=1,
                            weight_init='glorot', validate_gradient=True, epsilon=epsilon)
            perceptron.train(train_data=train_data, nb_classes=10)
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
