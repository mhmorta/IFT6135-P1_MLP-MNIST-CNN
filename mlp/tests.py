import unittest
import numpy as np
from .neuralnetwork import NN
from .utils import plot_decision_boundary
import gzip
import pickle
import time


class Test(unittest.TestCase):

    def test_moon_hidden_number(self):

        train_data, test_data = two_moon_dataset()
        perceptron = NN(epochs=1000, hidden_dims=[50, 20], mu=0.5, batch_size=10)
        perceptron.train(train_data)
        perceptron.save_state("moons.pkl")
        prediction = perceptron.compute_predictions(test_data[:, :-1])  # pass only the features without labels
        expected = test_data[:, -1].astype(int)  # labels
        print("\nError rate: ", 1 - np.mean(prediction == expected))
        plot_decision_boundary(perceptron, test_data)

    def test_moons_load_params(self):

        train_data, test_data = two_moon_dataset()
        perceptron = NN()
        perceptron.load_state("moons.pkl")
        prediction = perceptron.compute_predictions(test_data[:, :-1])  # pass only the features without labels
        expected = test_data[:, -1].astype(int)  # labels
        print("\nError rate: ", 1 - np.mean(prediction == expected))
        plot_decision_boundary(perceptron, test_data)

    def test_moon_validate_gradient(self):
        train_data, test_data = two_moon_dataset()
        perceptron = NN(epochs=1, hidden_dims=[50, 20], mu=0.05, validate_gradient=True)
        perceptron.train(train_data)

    def test_mnist(self):
        with gzip.open('mlp/datasets/mnist.pkl.gz') as f:
            # encoding='latin1' --> https://stackoverflow.com/a/41366785
            data = pickle.load(f, encoding='latin1')

        x_train = data[0][0]
        y_train = data[0][1]

        x_valid = data[1][0]
        y_valid = data[1][1]

        x_test = data[2][0]
        y_test = data[2][1]

        train_data = np.append(x_train, y_train[..., None], axis=1)
        validation_data = np.append(x_valid, y_valid[..., None], axis=1)
        test_data = np.append(x_test, y_test[..., None], axis=1)

        perceptron = NN(epochs=1, hidden_dims=[10, 20], mu=0.5, batch_size=50)
        perceptron.test_data = test_data
        perceptron.validation_data = validation_data

        start = time.time()
        perceptron.train(train_data)
        stop = time.time()
        print("Time: %s" % (stop - start))
        #perceptron.save_state("mnist.pkl")

        prediction = perceptron.compute_predictions(test_data[:, :-1])  # pass only the features without labels
        expected = test_data[:, -1].astype(int)  # labels
        print("\nError rate: ", 1 - np.mean(prediction == expected))


def two_moon_dataset():
    with open('mlp/datasets/2moons.txt', 'r') as f:
        data = np.loadtxt(f)
        np.random.shuffle(data)
        train_data, test_data = np.vsplit(data, 2)
    return train_data, test_data
