import numpy as np
from mlp.neuralnetwork import NN
import gzip
import pickle

import os
os.system("taskset -p 0xff %d" % os.getpid())

f = gzip.open('datasets/mnist.pkl.gz')
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

perceptron = NN(epochs=100, hidden_dims=[500, 500], mu=0.1, batch_size=100, debug=True)
perceptron.test_data = test_data
perceptron.validation_data = validation_data

perceptron.train(train_data)

perceptron.save_state("mnist.pkl")

prediction = perceptron.compute_predictions(test_data[:, :-1])  # pass only the features without labels
expected = test_data[:, -1].astype(int)  # labels
print("\nError rate: ", 1 - np.mean(prediction == expected))