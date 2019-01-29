import numpy as np
from .utils import random_uniform, safe_softmax_matrix, relu, onehot_matrix, one_hot, safe_softmax, relu_derivative
import pickle


class MLPerceptron(object):
    """
    Realisation of the 2-layer neural network that uses numpy matrix methods
    """

    def __init__(self, nb_hidden=6, mu=0.01, epochs=100, batch_size=50,
                 l11=0.005, l12=0.001, l21=0.005, l22=0.001, grad_threshold=0.02, validate_gradient=False, debug=False):
        self.nb_hidden = nb_hidden
        self.mu = mu
        self.epochs = epochs
        self.grad_threshold = grad_threshold
        self.nb_out = 0
        self.nb_samples = 0
        self.epsilon = 1e-5  # default gradient verification step
        self.train_data = None  # features from training data
        self.w2 = None
        self.w1 = None
        self.b1 = None
        self.b2 = None
        self._ha = None  # hidden layer activations
        self._hs = None  # hidden layer outputs
        self._out = None  # network output layer outputs
        self._validate_gradient = validate_gradient
        self.batch_size = batch_size
        self.l11 = l11
        self.l12 = l12
        self.l21 = l21
        self.l22 = l22
        self.report_file_name = "report.csv"
        self.validation_data = None
        self.test_data = None
        self.debug = debug

    def description(self):
        return "{0} : hidden={1}, learn.rate={2}, epochs={3}, batch={4}\n" \
               r"$\lambda11={5}, \lambda12={6}, \lambda21={7}, \lambda22={8}$".format(
            self.__class__.__name__,
            self.nb_hidden,
            self.mu,
            self.epochs,
            self.batch_size,
            self.l11,
            self.l12,
            self.l21,
            self.l22)

    def init_weights(self, i, j):
        """
        Initialize the weight matrix i x j for the layer of n entries
        :param i: number of rows of the matrix W
        :param j: number of columns
        :return: two-dimensional numpy array
        """
        # todo why max to max?
        max_val = 1 / np.sqrt(j)
        return np.array([[random_uniform(-max_val, max_val) for _ in range(j)] for _ in range(i)])

    def init_bias(self, i):
        return np.zeros(i)

    def train(self, train_data):
        # initialize the weight matrices and bias arrays
        self.train_data = train_data
        self.nb_samples = np.shape(self.train_data)[0]
        classes = train_data[:, -1].astype(int)  # convert labels to integers
        nb_features = np.shape(train_data)[1] - 1  # exclude the last column which contains the labels
        self.nb_out = np.unique(classes).size  # number neurons in the output layer == number of the classes
        self.w2 = self.init_weights(self.nb_out, self.nb_hidden)  # dimension m X dh
        self.w1 = self.init_weights(self.nb_hidden, nb_features)  # dimensions dh x X
        self.b2 = self.init_bias(self.nb_out)  # dimensions m
        self.b1 = self.init_bias(self.nb_hidden)  # dimension dh
        # train
        report = None
        if self.debug:
            report = open(self.report_file_name, 'w')
            report.write("epoch,train_error,train_avg_loss,valid_error,valid_avg_loss,test_error,test_avg_loss\n")
        for epoch in range(self.epochs):
            #
            if self.debug:
                self.evaluate_and_log(epoch, report)
            # split the data into mini-batches and train the network for the each batch
            np.random.shuffle(self.train_data)
            for i in range(0, self.nb_samples, self.batch_size):
                batch = self.train_data[i:i + self.batch_size]
                self.train_batch(batch)

        if report is not None:
            report.close()

    def evaluate_and_log(self, epoch, report_file):
        stats = [epoch]
        print("\nEpoch: ", epoch)
        for name, data in [('train', self.train_data), ('validation', self.validation_data), ('test', self.test_data)]:
            if data is not None:
                prediction = self.compute_predictions(data[:, :-1])  # pass only the features without labels
                expected = data[:, -1].astype(int)  # labels
                error = 1 - np.mean(prediction == expected)
                avg_loss = self.average_loss(expected)
                print("%s error rate: " % name, error)
                print("%s average loss: " % name, avg_loss)
                stats.append(error)
                stats.append(avg_loss)
            else:
                stats.append('')
                stats.append('')
        report_file.write(','.join(map(str, stats)) + '\n')

    def load_state(self, params_file):
        f = open(params_file, 'rb')
        state = pickle.load(f)
        for attr in state.items():
            self.__setattr__(attr[0], attr[1])

        print("\nLoaded classifier configuration ----------------------------")
        print("W1: ", self.w1)
        print("b1: ", self.b1)
        print("W2: ", self.w2)
        print("b2: ", self.b2)

    def save_state(self, params_file):
        attributes = {k: v for k, v in self.__dict__.items() if not k.startswith('__')}
        print("\nSaving classifier configuration ----------------------------")
        for param in attributes.items():
            print(param)
        f = open(params_file, 'wb')
        pickle.dump(attributes, f)

    def train_batch(self, batch):
        x = batch[:, :-1]
        y = batch[:, -1].astype(int)  # convert labels to integers
        self.fprop(x)
        # Gradient check for each parameter
        backprop_gradient = self.bprop(x, y)
        if self._validate_gradient:
            self.validate_gradient(x, y, backprop_gradient)
        self.update_parameters(*backprop_gradient)

    def validate_gradient(self, x, y, backprop_gradient):
        """
        Validate if the gradient calculated by backpropagation algorithm is similar
        to the empirical gradient calculated with finite step epsilon
        :param x: array of features
        :param y: array of labels
        :param backprop_gradient: gradient calculated by backpropagation algorithm
        :raises: exception if the calculated algorithm is too different from empirical
        """
        # calculate finite gradient
        model_parameters = ['w1', 'b1', 'w2', 'b2']
        for pidx, pname in enumerate(model_parameters):
            # Get the actual parameter value by it's name, e.g. w1, w2 etc
            parameter = self.__getattribute__(pname)
            # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
            #TODO ask?
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                # Save the original value so we can reset it later
                original_value = parameter[ix]
                # Estimate the gradient using (f(x+h) - f(x-h))/2h
                # calculate the empirical error for x+h
                parameter[ix] = original_value + self.epsilon
                self.fprop(x)
                grad_plus = self.empirical_error(y)
                # calculate the empirical error for x-h
                parameter[ix] = original_value - self.epsilon
                self.fprop(x)
                grad_minus = self.empirical_error(y)
                # Reset parameter to original value
                parameter[ix] = original_value

                # verify gradient
                estimated_gradient = (grad_plus - grad_minus) / (2 * self.epsilon)
                calculated_gradient = backprop_gradient[pidx][ix]
                diff = np.abs(calculated_gradient - estimated_gradient)
                summ = np.abs(calculated_gradient) + np.abs(estimated_gradient)
                if summ != 0:
                    grad_error = diff / summ
                    print("\nGradient error: ", grad_error)
                    if grad_error > self.grad_threshold:
                        print("\nEstimated gradient: ", estimated_gradient)
                        print("\nCalculated gradient: ", calculated_gradient)
                        raise Exception("\nGradient diff is too big: ", grad_error)
                it.iternext()

    def empirical_error(self, y):
        """
        Compute regularized empirical error
        :param y: numpy array of the expected classes
        :return:
        """
        prediction = np.multiply(self._out, onehot_matrix(self.nb_out, y))
        precision = np.max(prediction, axis=1)
        log_err = np.multiply(np.log(precision), -1)
        err = np.mean(log_err)
        regularized_err = err + \
                          self.l11 * abs(self.w1).sum() + \
                          self.l21 * abs(self.w2).sum() + \
                          self.l12 * (self.w1 ** 2).sum() + \
                          self.l22 * (self.w2 ** 2).sum()
        return regularized_err

    def average_loss(self, y):
        """
        Compute the average loss function
        :param y: numpy array of the expected classes
        :return:
        """
        prediction = np.multiply(self._out, onehot_matrix(self.nb_out, y))
        precision = np.max(prediction, axis=1)
        log_err = np.multiply(np.log(precision), -1)
        return np.mean(log_err)

    def fprop(self, x):
        """
        walk forward from input layer x to output layer
        """
        self._ha = np.dot(x, self.w1.transpose()) + self.b1
        self._hs = relu(self._ha)  # hidden layer output
        oa = np.dot(self._hs, self.w2.transpose()) + self.b2
        self._out = safe_softmax_matrix(oa)  # network output

    def bprop(self, x, y):
        """
        Backpropagation algorithm realisation for 2-layer network
        :param x: numpy array of the features
        :param y: numpy array of the expected classes
        :return:
        """
        # calculate gradients
        # start from the output layer
        grad_oa = self._out - onehot_matrix(self.nb_out, y)
        # apply regularisation (weight decay)
        grad_w2 = np.dot(grad_oa.transpose(), self._hs) / self.batch_size + \
                  np.multiply(np.sign(self.w2), self.l21) + \
                  np.multiply(self.w2, (self.l22 * 2))
        grad_b2 = np.sum(grad_oa, axis=0) / self.batch_size
        # then pass to the hidden layer
        grad_hs = np.dot(grad_oa, self.w2)
        grad_ha = np.multiply(grad_hs, relu_derivative(self._ha))
        # apply regularisation (weight decay)
        grad_w1 = np.dot(grad_ha.transpose(), x) / self.batch_size + \
                  np.multiply(np.sign(self.w1), self.l11) + \
                  np.multiply(self.w1, (self.l12 * 2))
        grad_b1 = np.sum(grad_ha, axis=0) / self.batch_size
        return grad_w1, grad_b1, grad_w2, grad_b2

    def update_parameters(self, grad_w1, grad_b1, grad_w2, grad_b2):
        # update network parameters W1, W2, b1 and b2
        self.w2 -= self.mu * grad_w2
        self.b2 -= self.mu * grad_b2
        self.w1 -= self.mu * grad_w1
        self.b1 -= self.mu * grad_b1

    def compute_predictions(self, test_data):
        # return the most probable class
        self.fprop(test_data)
        #todo give out only one element
        return np.argmax(self._out, axis=1)  # we assume that the index == class


class MLPerceptronIterative(object):
    """
    Realisation of the 2-layer neural network that uses iterative calculation of mini-batch
    """

    def __init__(self, nb_hidden=6, mu=0.01, epochs=100, batch_size=50,
                 l11=0.005, l12=0.001, l21=0.005, l22=0.001, grad_threshold=0.5, validate_gradient=False):
        self.nb_hidden = nb_hidden
        self.mu = mu
        self.epochs = epochs
        self.grad_threshold = grad_threshold
        self.nb_out = 0
        self.nb_features = 0
        self.nb_samples = 0
        self.epsilon = 1e-5  # default gradient verification step
        self.train_data = None  # features from training data
        self.w2 = None
        self.w1 = None
        self.b1 = None
        self.b2 = None
        self._ha = None  # hidden layer activations
        self._hs = None  # hidden layer outputs
        self._out = None  # network output layer outputs
        self._validate_gradient = validate_gradient
        self.batch_size = batch_size
        self.l11 = l11
        self.l12 = l12
        self.l21 = l21
        self.l22 = l22

    def description(self):
        return "{0} : hidden={1}, learn.rate={2}, epochs={3}, batch={4}\n" \
               r"$\lambda11={5}, \lambda12={6}, \lambda21={7}, \lambda22={8}$".format(
            self.__class__.__name__,
            self.nb_hidden,
            self.mu,
            self.epochs,
            self.batch_size,
            self.l11,
            self.l12,
            self.l21,
            self.l22)

    def init_weights(self, i, j):
        """
        Initialize the weight matrix i x j for the layer of n entries
        :param i: number of rows of the matrix W
        :param j: number of columns
        :return: two-dimensional numpy array
        """
        max_val = 1 / np.sqrt(j)
        return np.matrix([[random_uniform(-max_val, max_val) for _ in range(j)] for _ in range(i)])

    @staticmethod
    def init_bias(i):
        return np.zeros(i)

    def train(self, train_data):
        # initialize the weight matrices and bias arrays
        self.train_data = train_data
        self.nb_samples = np.shape(self.train_data)[0]
        classes = train_data[:, -1].astype(int)  # convert labels to integers
        self.nb_features = np.shape(train_data)[1] - 1  # exclude the last column which contains the labels
        self.nb_out = np.max(classes) + 1  # number neurons in the output layer == number of the classes
        self.w2 = self.init_weights(self.nb_out, self.nb_hidden)  # dimension m X dh
        self.w1 = self.init_weights(self.nb_hidden, self.nb_features)  # dimensions dh x X
        self.b2 = self.init_bias(self.nb_out)  # dimensions m
        self.b1 = self.init_bias(self.nb_hidden)  # dimension dh
        # train
        for epoch in range(self.epochs):
            # split the data into mini-batches and train the network for the each batch
            np.random.shuffle(self.train_data)
            for i in range(0, self.nb_samples, self.batch_size):
                batch = self.train_data[i:i + self.batch_size]
                self.train_batch(batch)
        print("\nCalculated coefficients:")
        print("\nW1: %s" % self.w1)
        print("\nb1: %s" % self.b1)
        print("\nW2: %s" % self.w2)
        print("\nb2: %s" % self.b2)

    def train_batch(self, batch):
        batch_size = len(batch)
        batch_gradient_w1 = np.zeros((self.nb_hidden, self.nb_features))
        batch_gradient_w2 = np.zeros((self.nb_out, self.nb_hidden))
        batch_gradient_b1 = np.zeros((1, self.nb_hidden))
        batch_gradient_b2 = np.zeros((1, self.nb_out))
        for example in batch:
            x = example[:-1]  # array, dim = d, # of features, or input neurones
            y = int(example[-1])  # scalar
            self.fprop(x)
            # Gradient check for each parameter
            grad_w1, grad_b1, grad_w2, grad_b2 = self.bprop(x, y)

            # accumulate gradients
            batch_gradient_w1 += grad_w1
            batch_gradient_b1 += grad_b1
            batch_gradient_w2 += grad_w2
            batch_gradient_b2 += grad_b2

        grad_w1 = (batch_gradient_w1 +
                   np.multiply(np.sign(self.w1), self.l11) +
                   np.multiply(self.w1, (self.l12 * 2))) / batch_size

        # grad_w1 = batch_gradient_w1 / batch_size

        grad_b1 = batch_gradient_b1 / batch_size

        grad_w2 = (batch_gradient_w2 +
                   np.multiply(np.sign(self.w2), self.l21) +
                   np.multiply(self.w2, (self.l22 * 2))) / batch_size

        # grad_w2 = batch_gradient_w2 / batch_size

        grad_b2 = batch_gradient_b2 / batch_size

        batch_x = batch[:, :-1]
        batch_y = batch[:, -1].astype(int)

        if self._validate_gradient:
            self.validate_gradient(batch_x, batch_y, grad_w1, grad_b1, grad_w2, grad_b2)

        self.update_parameters(grad_w1, grad_b1, grad_w2, grad_b2)

    def validate_gradient(self, x, y, grad_w1, grad_b1, grad_w2, grad_b2):
        """
        Validate if the gradient calculated by backpropagation algorithm is similar
        to the empirical gradient calculated with finite step epsilon
        :param x: array of features, dim = K x d
        :param y: array of labels, dim = K
        :raises: exception if the calculated algorithm is too different from empirical
        """
        # calculate finite gradient
        model_parameters = ['w1', 'b1', 'w2', 'b2']
        backprop_gradient = [grad_w1, grad_b1, grad_w2, grad_b2]
        for pidx, pname in enumerate(model_parameters):
            # Get the actual parameter value by it's name, e.g. w1, w2 etc
            parameter = self.__getattribute__(pname)
            # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                # Save the original value so we can reset it later
                original_value = parameter[ix]
                # Estimate the gradient using (f(x+h) - f(x-h))/2h
                # calculate the empirical error for x+h
                parameter[ix] = original_value + self.epsilon
                self.fprop(x)
                grad_plus = self.empirical_error(y)
                # calculate the empirical error for x-h
                parameter[ix] = original_value - self.epsilon
                self.fprop(x)
                grad_minus = self.empirical_error(y)
                # Reset parameter to original value
                parameter[ix] = original_value

                # verify gradient
                estimated_gradient = (grad_plus - grad_minus) / (2 * self.epsilon)
                calculated_gradient = backprop_gradient[pidx][ix]
                diff = np.abs(calculated_gradient - estimated_gradient)
                summ = np.abs(calculated_gradient) + np.abs(estimated_gradient)
                grad_error = 0
                if summ != 0:
                    grad_error = diff / summ
                print("\ngradient %s%s: estimated: %s, calculated: %s, diff: %s" % (
                    pname, ix, estimated_gradient, calculated_gradient, grad_error))
                if grad_error > self.grad_threshold:
                    raise Exception("\nGradient diff is too big: ", grad_error)
                it.iternext()

    def empirical_error(self, y):
        """
        Compute regularized empirical error
        :param y: numpy array of the expected classes
        :return:
        """
        prediction = np.multiply(self._out, onehot_matrix(self.nb_out, y))
        precision = np.max(prediction, axis=1)
        log_err = np.multiply(np.log(precision), -1)
        err = np.mean(log_err)
        regularized_err = err + \
                          self.l11 * abs(self.w1).sum() + \
                          self.l21 * abs(self.w2).sum() + \
                          self.l12 * np.power(self.w1, 2).sum() + \
                          self.l22 * np.power(self.w2, 2).sum()
        return regularized_err

    def fprop(self, x):
        """
        walk forward from input layer x to output layer
        """
        # apply W1 weights + B1 bias
        # vector ha: dim = dh
        self._ha = np.dot(x, self.w1.transpose()) + self.b1

        # apply non-linear activation function RELU
        # vector hs: dim = dh
        self._hs = relu(self._ha)

        # apply W2 weights + B2 bias
        # vector oa: dim = m
        oa = np.dot(self._hs, self.w2.transpose()) + self.b2

        # apply softmax function
        # out dimension: m (# number of output neurones)
        self._out = safe_softmax(oa)

    def bprop(self, x, y):
        """
        Calculate gradients using backpropagation algorithm
        :param x: numpy array of the features, dim = d
        :param y: expected class (scalar)
        :return:
        """
        # calculate gradients
        # start from the output layer
        # vector grad_oa: dim = m (# of out neurons)
        grad_oa = self._out - one_hot(self.nb_out, y)

        # gradient for W2
        # vector hs : out of the hidden layer, dim = dh
        # matrix grad_w2 : dim = dh x m
        grad_w2 = np.dot(grad_oa.transpose(), self._hs)

        # gradient for B2
        # vector grad_b2 : dim = m
        grad_b2 = grad_oa

        # gradient for W1
        # vector grad_hs : dim = dh
        grad_hs = np.dot(grad_oa, self.w2)
        # vector grad_ha : dim = dh
        grad_ha = np.multiply(grad_hs, relu_derivative(self._ha))
        # matrix grad_w1 : dim = dh x d
        grad_w1 = np.dot(grad_ha.transpose(), np.array([x]))
        # vector grad_b1: dim = dh (# of hidden neurons)
        grad_b1 = grad_ha
        return grad_w1, grad_b1, grad_w2, grad_b2

    def update_parameters(self, grad_w1, grad_b1, grad_w2, grad_b2):
        # update network parameters W1, W2, b1 and b2
        self.w1 -= self.mu * grad_w1
        self.b1 -= self.mu * grad_b1
        self.w2 -= self.mu * grad_w2
        self.b2 -= self.mu * grad_b2

    def compute_predictions(self, test_data):
        # return the most probable class
        self.fprop(test_data)
        return np.argmax(self._out, axis=1)  # we assume that the index == class
