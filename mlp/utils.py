import numpy as np
from matplotlib import pyplot as plt


# todo why safe?
def safe_softmax(v):
    # v is a vector
    exps = np.exp(v - np.max(v))
    res = exps / np.sum(exps)
    return res


def safe_softmax_matrix(m):
    return np.array([safe_softmax(v) for v in m])


def random_uniform(min_val, max_val):
    return np.random.uniform(min_val, max_val, 1)[0]


def relu(x):
    return np.maximum(x, 0)


def onehot_matrix(m, y):
    targets = np.array([y]).reshape(-1)
    return np.eye(m)[targets]


def one_hot(m, y):
    return np.array(map(lambda x: 1 if x == y else 0, range(m)))


def relu_derivative(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


def plot_decision_boundary(classifier, train_data):
    """
    inspired by : https://gist.github.com/dennybritz/ff8e7c2954dd47a4ce5f
    """
    font = {'family': 'Sans',
            'weight': 'bold',
            'size': 8}

    plt.rc('font', **font)
    # Set min and max values and give it some padding
    x_min, x_max = train_data[:, 0].min() - .5, train_data[:, 0].max() + .5
    y_min, y_max = train_data[:, 1].min() - .5, train_data[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = classifier.compute_predictions(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral_r)
    plt.scatter(train_data[:, 0], train_data[:, 1], c=train_data[:, -1], cmap=plt.cm.PRGn)
    plt.title(classifier.description())
    plt.show()


def test():
    print(safe_softmax_matrix(np.array([[-0.07843759, -0.24158356, 0.32002114]])))
    print(onehot_matrix(3, np.array([1])))
    print(one_hot(3, 1))
    print(relu([-1, 2]))
    print(relu_derivative(np.array([-0.07843759, -0.24158356,  0.32002114])))
    print(np.reshape([5, 6] * np.matrix([[1, 2]]).transpose(), 1))
