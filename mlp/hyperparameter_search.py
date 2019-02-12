import numpy as np
from mlp.neuralnetwork import NN
from mlp.utils import load_mnist_data


import os
os.system("taskset -p 0xff %d" % os.getpid())

train_data, validation_data, test_data = load_mnist_data()

combinations = [(4.614313552623348e-05, [500, 600]), (4.614313552623348e-05, [500, 500]), (4.614313552623348e-05, [600, 700]),
 (4.614313552623348e-05, [700, 500]), (0.1468423686421048, [500, 600]), (0.1468423686421048, [500, 500]),
 (0.1468423686421048, [600, 700]), (0.1468423686421048, [700, 500]), (0.3112553202446868, [500, 600]),
 (0.3112553202446868, [500, 500]), (0.3112553202446868, [600, 700]), (0.3112553202446868, [700, 500]),
 (4.311580209552355e-06, [500, 600]), (4.311580209552355e-06, [500, 500]), (4.311580209552355e-06, [600, 700]),
 (4.311580209552355e-06, [700, 500]), (0.013249755451138266, [500, 600]), (0.013249755451138266, [500, 500]),
 (0.013249755451138266, [600, 700]), (0.013249755451138266, [700, 500])]

sample_ids = np.random.choice(len(combinations), 10, replace=False)
for s_id in sample_ids:
    mu, h_d = combinations[s_id]
    perceptron = NN(epochs=10, hidden_dims=h_d, mu=mu, batch_size=32, weight_init='glorot',
                    debug=True)
    perceptron.test_data = test_data
    perceptron.validation_data = validation_data
    perceptron.train(train_data=train_data, nb_classes=10)

    perceptron.save_state("checkpoints/checkpoint_{}.pkl".format(perceptron.params_str()))
