import numpy as np
from mlp.neuralnetwork import NN
from mlp.utils import load_mnist_data


import os
os.system("taskset -p 0xff %d" % os.getpid())

train_data, validation_data, test_data = load_mnist_data()

perceptron = NN(epochs=10, hidden_dims=[500, 600], mu=0.01, batch_size=32, weight_init='zeros',
                debug=True)
perceptron.test_data = test_data
perceptron.validation_data = validation_data
perceptron.train(train_data=train_data, nb_classes=10)

perceptron.save_state("checkpoints/checkpoint_{}.pkl".format(perceptron.params_str()))

prediction = perceptron.test(test_data[:, :-1])  # pass only the features without labels
expected = test_data[:, -1].astype(int)  # labels
print("\nTest error rate: ", 1 - np.mean(prediction == expected))
print("\nTest accuracy: ", np.mean(prediction == expected) * 100)