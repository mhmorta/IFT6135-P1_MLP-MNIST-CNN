import numpy as np
from .utils import random_uniform, safe_softmax_matrix, relu, onehot_matrix, one_hot, safe_softmax, relu_derivative
import pickle


class NN:
    """
    Realisation of the 2-layer neural network that uses numpy matrix methods
    """

    def __init__(self, hidden_dims=(500, 500), mu=0.01, epochs=100, batch_size=50, grad_threshold=0.02, validate_gradient=False, debug=False, param_init='normal'):
        self.nb_hidden1, self.nb_hidden2 = hidden_dims
        self.mu = mu
        self.epochs = epochs
        self.grad_threshold = grad_threshold
        self.nb_out = 0
        self.nb_samples = 0
        self.epsilon = 1e-5  # default gradient verification step
        self.train_data = None  # features from training data
        self.param_init = param_init

        self.w3 = None
        self.w2 = None
        self.w1 = None

        self.b1 = None
        self.b2 = None
        self.b3 = None

        self._ha1 = None  # hidden layer 1 activations
        self._hs1 = None  # hidden layer 1 outputs

        self._ha2 = None  # hidden layer 1 activations
        self._hs2 = None  # hidden layer 1 outputs

        self._out = None  # network output layer outputs
        self._validate_gradient = validate_gradient

        self.batch_size = batch_size

        self.report_file_name = "report.csv"
        self.validation_data = None
        self.test_data = None
        self.debug = debug
        print('Using weight initialization:', param_init)

    def description(self):
        return "{0} : nb_hidden1={1}, nb_hidden2={2}, learn.rate={3}, epochs={4}, batch={5}\n".format(
            self.__class__.__name__,
            self.nb_hidden1,
            self.nb_hidden2,
            self.mu,
            self.epochs,
            self.batch_size)

    def initialize_weights(self, i, j):
        """
        Initialize the weight matrix i x j for the layer of n entries
        :param i: number of rows of the matrix W
        :param j: number of columns
        :return: two-dimensional numpy array
        """
        if self.param_init == 'working':
            max_val = 1 / np.sqrt(j)
            return np.array([[random_uniform(-max_val, max_val) for _ in range(j)] for _ in range(i)])
        elif self.param_init == 'zeros':
            weights = np.zeros((i, j))
        elif self.param_init == 'normal':
            weights = np.array([[np.random.normal(0, 1) for _ in range(j)] for _ in range(i)])
        elif self.param_init == 'glorot':
            d = np.sqrt(6 / (i + j))
            weights = np.array([[random_uniform(-d, d) for _ in range(j)] for _ in range(i)])
        return weights

    def init_bias(self, i):
        return np.zeros(i)

    def train(self, train_data):
        # initialize the weight matrices and bias arrays
        self.train_data = train_data
        self.nb_samples = np.shape(self.train_data)[0]
        classes = train_data[:, -1].astype(int)  # convert labels to integers
        nb_features = np.shape(train_data)[1] - 1  # exclude the last column which contains the labels
        self.nb_out = np.unique(classes).size  # number neurons in the output layer == number of the classes
        self.w3 = self.initialize_weights(self.nb_out, self.nb_hidden2)
        self.w2 = self.initialize_weights(self.nb_hidden2, self.nb_hidden1)
        self.w1 = self.initialize_weights(self.nb_hidden1, nb_features)
        self.b3 = self.init_bias(self.nb_out)
        self.b2 = self.init_bias(self.nb_hidden2)
        self.b1 = self.init_bias(self.nb_hidden1)
        # train
        report = None
        with open(self.report_file_name, 'w') as report:
            if self.debug:
                report.write("epoch,train_error,train_avg_loss,valid_error,valid_avg_loss,test_error,test_avg_loss\n")
            for epoch in range(self.epochs):
                if self.debug:
                    self.evaluate_and_log(epoch, report)
                # split the data into mini-batches and train the network for the each batch
                #todo return shuffle
                #np.random.shuffle(self.train_data)
                for i in range(0, self.nb_samples, self.batch_size):
                    batch = self.train_data[i:i + self.batch_size]
                    self.train_batch(batch)

    def evaluate_and_log(self, epoch, report_file):
        stats = [epoch]
        print("\nEpoch: ", epoch)
        for name, data in [('train', self.train_data), ('validation', self.validation_data)]:
            #data: n X 784 + 1
            if data is not None:
                prediction = self.compute_predictions(data[:, :-1])  # pass only the features without labels
                expected = data[:, -1].astype(int)  # labels
                error = 1 - np.mean(prediction == expected)
                avg_loss = self.average_loss(expected)
                print("%s error rate: " % name, error)
                print("%s average loss: " % name, avg_loss)
                print("%s accuracy: " % name, (1 - error) * 100)
                stats.append(error)
                stats.append(avg_loss)
            else:
                stats.append('')
                stats.append('')
        report_file.write(','.join(map(str, stats)) + '\n')

    def load_state(self, params_file):
        with open(params_file, 'rb') as f:
            state = pickle.load(f)
        for attr in state.items():
            self.__setattr__(attr[0], attr[1])

        print("\nLoaded classifier configuration ----------------------------")
        print("W1: ", self.w1)
        print("b1: ", self.b1)
        print("W2: ", self.w2)
        print("b2: ", self.b2)
        print("W3: ", self.w3)
        print("b3: ", self.b3)

    def save_state(self, params_file):
        attributes = {k: v for k, v in self.__dict__.items() if not k.startswith('__')}
        print("\nSaving classifier configuration ----------------------------")
        for param in attributes.items():
            print(param)
        with open(params_file, 'wb') as f:
            pickle.dump(attributes, f)

    def train_batch(self, batch):
        x = batch[:, :-1]
        y = batch[:, -1].astype(int)  # convert labels to integers
        self.forward(x)
        # Gradient check for each parameter
        backprop_gradient = self.backward(x, y)
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
        model_parameters = ['w1', 'b1', 'w2', 'b2', 'w3', 'b3']
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
                self.forward(x)
                grad_plus = self.empirical_error(y)
                # calculate the empirical error for x-h
                parameter[ix] = original_value - self.epsilon
                self.forward(x)
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
                    if grad_error > self.grad_threshold:
                        #todo shift up 'Gradient error'
                        print("\nGradient error: ", grad_error)
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
        return err

    def average_loss(self, y):
        """
        Compute the average loss function
        :param y: numpy array of the expected classes [5, 0, 4, ...] of size n
        :return:
        """
        # self._out: 5000 x 10, onehot_matrix: 5000 x 10
        prediction = np.multiply(self._out, onehot_matrix(self.nb_out, y))
        precision = np.max(prediction, axis=1)
        # https://stackoverflow.com/a/52209380
        loggg = np.log2(precision)
        #loggg = np.log2(precision, out=np.zeros_like(precision), where=(precision != 0))
        log_err = np.multiply(loggg, -1)
        return np.mean(log_err)

    def forward(self, x):
        """
        walk forward from input layer x to output layer
        """
        self._ha1 = np.dot(x, self.w1.transpose()) + self.b1  # first hidden layer activation
        self._hs1 = self.activation(self._ha1)  # first hidden layer output

        self._ha2 = np.dot(self._hs1, self.w2.transpose()) + self.b2 # second hidden layer activation
        self._hs2 = self.activation(self._ha2)  # second hidden layer output

        oa = np.dot(self._hs2, self.w3.transpose()) + self.b3 # output layer activation
        self._out = self.softmax(oa)  # network output

    def activation(self, input):
        return relu(input)

    def softmax(self, input):
        return safe_softmax_matrix(input)

    def backward(self, x, y):
        """
        Backpropagation algorithm realisation for 3-layer network
        :param x: numpy array of the features
        :param y: numpy array of the expected classes
        :return:
        """
        # calculate gradients
        # start from the output layer
        grad_oa = self._out - onehot_matrix(self.nb_out, y)

        grad_w3 = np.dot(grad_oa.transpose(), self._hs2) / self.batch_size
        grad_b3 = np.sum(grad_oa, axis=0) / self.batch_size

        # then pass to the second hidden layer
        grad_hs2 = np.dot(grad_oa, self.w3)
        grad_ha2 = np.multiply(grad_hs2, relu_derivative(self._ha2))

        grad_w2 = np.dot(grad_ha2.transpose(), self._hs1) / self.batch_size
        grad_b2 = np.sum(grad_ha2, axis=0) / self.batch_size

        # then pass to the first hidden layer
        grad_hs1 = np.dot(grad_ha2, self.w2)
        grad_ha1 = np.multiply(grad_hs1, relu_derivative(self._ha1))

        grad_w1 = np.dot(grad_ha1.transpose(), x) / self.batch_size
        grad_b1 = np.sum(grad_ha1, axis=0) / self.batch_size

        return grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3

    def update_parameters(self, grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3):
        # update network parameters W1, W2, b1 and b2
        self.w3 -= self.mu * grad_w3
        self.b3 -= self.mu * grad_b3
        self.w2 -= self.mu * grad_w2
        self.b2 -= self.mu * grad_b2
        self.w1 -= self.mu * grad_w1
        self.b1 -= self.mu * grad_b1

    def compute_predictions(self, test_data):
        # return the most probable class
        self.forward(test_data)
        #todo give out only one element
        return np.argmax(self._out, axis=1)  # we assume that the index == class



