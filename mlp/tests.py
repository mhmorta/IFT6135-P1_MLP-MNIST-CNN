import unittest
import numpy as np
from .neuralnetwork import MLPerceptron, MLPerceptronIterative
from .utils import plot_decision_boundary
import gzip
import pickle
import time


class Test(unittest.TestCase):

    def test_moon_hidden_number(self):

        train_data, test_data = two_moon_dataset()
        perceptron = MLPerceptron(epochs=1000, nb_hidden=50, mu=0.5, batch_size=10, l11=5e-5, l12=1e-5, l21=5e-5, l22=1e-5)
        perceptron.train(train_data)
        perceptron.save_state("moons.pkl")
        prediction = perceptron.compute_predictions(test_data[:, :-1])  # pass only the features without labels
        expected = test_data[:, -1].astype(int)  # labels
        print("\nError rate: ", 1 - np.mean(prediction == expected))
        plot_decision_boundary(perceptron, test_data)

    def test_moons_load_params(self):

        train_data, test_data = two_moon_dataset()
        perceptron = MLPerceptron()
        perceptron.load_state("moons.pkl")
        prediction = perceptron.compute_predictions(test_data[:, :-1])  # pass only the features without labels
        expected = test_data[:, -1].astype(int)  # labels
        print("\nError rate: ", 1 - np.mean(prediction == expected))
        plot_decision_boundary(perceptron, test_data)

    def test_moon_regularisation(self):

        for l11, l12, l21, l22 in [(0.0003, 0.0001, 0.0003, 0.0001),
                                   (0.0005, 0.0001, 0.0005, 0.0001),
                                   (0.0005, 0.0001, 0.0003, 0.0001),
                                   (0.0003, 0.0001, 0.0005, 0.0001),
                                   (0.0007, 0.0002, 0.0003, 0.0001)]:
            train_data, test_data = two_moon_dataset()
            perceptron = MLPerceptron(epochs=10000, nb_hidden=50, mu=0.01, batch_size=50, l11=l11, l12=l12, l21=l21, l22=l22)
            perceptron.train(train_data)
            prediction = perceptron.compute_predictions(test_data[:, :-1])  # pass only the features without labels
            expected = test_data[:, -1].astype(int)  # labels
            print("\nError rate: ", 1 - np.mean(prediction == expected))
            plot_decision_boundary(perceptron, test_data)

    def test_moon_validate_gradient(self):
        train_data, test_data = two_moon_dataset()
        perceptron = MLPerceptronIterative(epochs=1, nb_hidden=4, mu=0.05, validate_gradient=True)
        perceptron.train(train_data)

    def test_mnist(self):
        f = gzip.open('mnist.pkl.gz')
        data = pickle.load(f)

        x_train = data[0][0]
        y_train = data[0][1]

        x_valid = data[1][0]
        y_valid = data[1][1]

        x_test = data[2][0]
        y_test = data[2][1]

        train_data = np.append(x_train, y_train[..., None], axis=1)
        validation_data = np.append(x_valid, y_valid[..., None], axis=1)
        test_data = np.append(x_test, y_test[..., None], axis=1)

        perceptron = MLPerceptron(epochs=1, nb_hidden=256, mu=0.5, batch_size=50, l11=5e-5, l12=1e-5, l21=5e-5, l22=1e-5)
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

    def test_moon_validate_gradient_iterative(self):
        train_data, test_data = two_moon_dataset()
        perceptron = MLPerceptronIterative(epochs=4, nb_hidden=2, mu=0.05, validate_gradient=True)
        perceptron.train(train_data)


def two_moon_dataset():
    data = np.loadtxt(open('2moons.txt', 'r'))
    np.random.shuffle(data)
    train_data, test_data = np.vsplit(data, 2)
    return train_data, test_data
