import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def relu(x, deriv=False):
    if deriv:
        return 1 * (x > 0)
    return x * (x > 0)


def sigmoid(x, deriv=False):
    if deriv:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + np.exp(-x))


def pre_process_dataset(path):
    train = pd.read_csv(path)

    # extract the output and convert into one-hot vector
    y = np.zeros((train.shape[0], 10))
    y[np.arange(train.shape[0]), train['label'].values] = 1

    # extract the features and scale the pixel value
    X = train.copy()
    X.drop(['label'], axis=1, inplace=True)
    X = X.values
    X = X / 255

    # split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

    # reshape the dataset
    X_train = X_train.T
    X_test = X_test.T
    y_train = y_train.T
    y_test = y_test.T

    return X_train, X_test, y_train, y_test


class NeuralNetwork(object):
    W = []
    b = []
    A = []
    Z = []
    size = 0


    def __init__(self, layer):
        self.W = [np.random.randn(j, i) * 0.01 for i, j in zip(layer[:-1], layer[1:])]
        self.b = [np.zeros((i, 1)) for i in layer[1:]]
        self.size = len(layer) - 1


    def feed_forward(self, X):
        self.A = []
        self.Z = []

        # append the X as A0 and Z0 (padding)
        self.A.append(X)
        self.Z.append(X)

        for i in range(self.size):
            z = np.dot(self.W[i], self.A[-1]) + self.b[i]

            # relu or sigmoid
            if i == self.size - 1:
                a = sigmoid(z)
            else:
                a = relu(z)

            self.Z.append(z)
            self.A.append(a)

        return self.A[-1]


    def calculate_cost(self, AL, y):
        m = y.shape[1]
        loss = np.sum(y * np.log(AL) + (1-y) * np.log(1-AL))
        return loss / (-m)


    def backward_propagation(self, AL, y):
        m = y.shape[1]
        grads = {}

        for i in reversed(range(self.size)):
            # last layer
            if i == self.size - 1:
                grads["dZ" + str(i + 1)] = AL - y
            else:
                grads["dZ" + str(i + 1)] = np.dot(self.W[i + 1].T, grads["dZ" + str(i + 2)]) * relu(self.Z[i+1], deriv=True)

            grads["dW" + str(i + 1)] = np.dot(grads["dZ" + str(i + 1)], self.A[i].T) / m
            grads["db" + str(i + 1)] = np.sum(grads["dZ" + str(i + 1)], axis=1, keepdims=True) / m

        return grads


    def fit(self, X, y, learning_rate, num_iterations):
        for _ in range(num_iterations):
            # feed forward
            prediction = self.feed_forward(X)

            # error calculation
            error = self.calculate_cost(prediction, y)
            print("Iteration: {}  |  Error: {}".format(_, error))

            # backpropagation
            grads = self.backward_propagation(prediction, y)

            # update parameters
            for i in range(len(self.W)):
                self.W[i] -= learning_rate * grads["dW" + str(i + 1)]
                self.b[i] -= learning_rate * grads["db" + str(i + 1)]
        return 1


    def evaluate(self, test_data, test_labels):
        prediction = self.feed_forward(test_data)
        prediction = np.argmax(prediction, axis=0)
        true_label = np.argmax(test_labels, axis=0)

        accuracy = np.sum(prediction == true_label) / test_labels.shape[1]

        return accuracy


if __name__ == "__main__":
    np.random.seed(23)

    X_train, X_test, y_train, y_test = pre_process_dataset("train.csv")

    model = NeuralNetwork([784, 150, 10])
    model.fit(X_train, y_train, learning_rate=0.3, num_iterations=100)

    accuracy = model.evaluate(X_test, y_test)
    print("Accuracy: ", accuracy)