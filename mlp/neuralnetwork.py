import numpy as np
from .utils import safe_softmax_matrix, relu, onehot_matrix, relu_derivative
import pickle


class NN:
    """
    Realisation of the 2-layer neural network that uses numpy matrix methods
    """

    def __init__(self, hidden_dims=(500, 500), mu=None, epochs=100, batch_size=50, grad_threshold=0.02, validate_gradient=False, debug=False, weight_init='glorot', epsilon = 1e-5):
        self.nb_hidden1, self.nb_hidden2 = hidden_dims
        self.mu = mu
        self.epochs = epochs
        self.grad_threshold = grad_threshold
        self.nb_out = 0
        self.nb_samples = 0
        self.epsilon = epsilon  # default gradient verification step
        self.train_data = None  # features from training data
        self.weight_init = weight_init

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
        print('======')
        print('weight_init:', weight_init)
        print('mu:', mu)
        print('hidden_dims:', hidden_dims)
        print('======')

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
        if self.weight_init == 'zeros':
            weights = np.zeros((i, j))
        elif self.weight_init == 'normal':
            weights = np.array(np.random.normal(0, 1, (i, j)))
        elif self.weight_init == 'glorot':
            d = np.sqrt(6 / (i + j))
            weights = np.random.uniform(-d, d, (i, j))
        else:
            # normal random initialization by default if not specified
            weights = np.array(np.random.normal(0, 1, (i, j)))
        return weights

    def init_bias(self, i):
        return np.zeros(i)

    def train(self, train_data):
        # initialize the weight matrices and bias arrays
        self.train_data = train_data
        #50 000
        self.nb_samples = np.shape(self.train_data)[0]
        # 10
        classes = train_data[:, -1].astype(int)  # convert labels to integers
        # 784
        nb_features = np.shape(train_data)[1] - 1  # exclude the last column which contains the labels
        # 10
        self.nb_out = np.unique(classes).size  # number neurons in the output layer == number of the classes
        # 10x200
        self.w3 = self.initialize_weights(self.nb_out, self.nb_hidden2)
        # 200x1000
        self.w2 = self.initialize_weights(self.nb_hidden2, self.nb_hidden1)
        # 1000x784
        self.w1 = self.initialize_weights(self.nb_hidden1, nb_features)
        # 10
        self.b3 = self.init_bias(self.nb_out)
        # 200
        self.b2 = self.init_bias(self.nb_hidden2)
        # 1000
        self.b1 = self.init_bias(self.nb_hidden1)
        # train
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
            # data: 50000 X 784 + 1
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

            # validate gradient for the second layer only
            if pname not in {'w2', 'b2'}: continue

            # Get the actual parameter value by it's name, e.g. w1, w2 etc
            parameter = self.__getattribute__(pname)

            # Iterate over each element of the parameter matrix,
            # method returns indexes (i, j) of each element e.g. (0,0), (0,1), ...
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])

            # verify only 10 parameters
            verified = 0
            while not it.finished and verified < 10:

                verified += 1

                # ix = (i, j)
                ix = it.multi_index

                # Save the original value so we can reset it later
                original_value = parameter[ix]

                # Estimate the gradient using (f(x+h) - f(x-h))/2h
                # calculate the empirical error for x+h
                parameter[ix] = original_value + self.epsilon
                self.forward(x)
                grad_plus = self.average_loss(y)

                # calculate the empirical error for x-h
                parameter[ix] = original_value - self.epsilon
                self.forward(x)
                grad_minus = self.average_loss(y)

                # Reset parameter to original value
                parameter[ix] = original_value

                # verify gradient
                estimated_gradient = (grad_plus - grad_minus) / (2 * self.epsilon)
                calculated_gradient = backprop_gradient[pidx][ix]
                diff = np.abs(calculated_gradient - estimated_gradient)
                print(pname + " gradient diff: ", diff)

                # throw an error if the gradient diff is too big
                summ = np.abs(calculated_gradient + estimated_gradient)

                grad_error = diff / summ if summ != 0 else 0
                if grad_error > self.grad_threshold:
                    print("------------- error ---------------")
                    print(pname + " gradient error: ", grad_error)
                    print(pname + " estimated gradient: ", estimated_gradient)
                    print(pname + " calculated gradient: ", calculated_gradient)
                    raise Exception("gradient error %f is above threshold %f" % (grad_error, self.grad_threshold))
                it.iternext()

    def average_loss(self, y):
        """
        Compute cross-entropy (log-loss) loss function
        :param y: numpy array of the expected classes
        :return:
        """
        prediction = np.multiply(self._out, onehot_matrix(self.nb_out, y))
        precision = np.max(prediction, axis=1)
        # safe log, will return 0 for log(0)
        log_precision = np.log(precision, out=np.zeros_like(precision), where=(precision != 0))
        log_err = np.multiply(log_precision, -1)
        err = np.mean(log_err)
        return err

    def forward(self, x):
        """
        walk forward from input layer x to output layer
        """
        # x: 50000 x 784
        # w1: 1000 x 784
        # b1: 1000
        # _ha1: 50000 x 1000
        # _hs1: 50000 x 1000
        self._ha1 = np.dot(x, self.w1.transpose()) + self.b1  # first hidden layer activation
        self._hs1 = self.activation(self._ha1)  # first hidden layer output
        # w2: 200 x 1000
        # b2: 200
        # _ha2: 50000 x 200
        # _hs2: 50000 x 200
        self._ha2 = np.dot(self._hs1, self.w2.transpose()) + self.b2  # second hidden layer activation
        self._hs2 = self.activation(self._ha2)  # second hidden layer output
        # 50000 x 10
        oa = np.dot(self._hs2, self.w3.transpose()) + self.b3  # output layer activation
        # 5000 x 10
        self._out = self.softmax(oa)  # network output
        if np.isnan(self._out).any():
            print('nans, put a debug point here')

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
        # todo give out only one element
        return np.argmax(self._out, axis=1)  # we assume that the index == class
