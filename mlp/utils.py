import numpy as np
import gzip
import pickle


def safe_softmax(v):
    exps = np.exp(v - np.max(v))
    res = exps / np.sum(exps)
    return res


def safe_softmax_matrix(m):
    return np.array([safe_softmax(v) for v in m])


def relu(x):
    return np.maximum(x, 0)


def onehot_matrix(m, y):
    targets = np.array([y]).reshape(-1)
    return np.eye(m)[targets]


def one_hot(m, y):
    return np.array(map(lambda x: 1 if x == y else 0, range(m)))


def relu_derivative(x):
    try:
        with np.errstate(all='raise'):
            x[x <= 0] = 0
            x[x > 0] = 1
    except Exception:
        print('Warning detected. Could be caused by high learning rate. x:', x)
    return x


def load_mnist_data():
    with gzip.open('datasets/mnist.pkl.gz') as f:
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

    return train_data, validation_data, test_data
