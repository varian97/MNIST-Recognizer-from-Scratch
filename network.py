import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
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


def update_with_gd(parameters, grads, learning_rate):
    l = len(parameters) // 2

    for i in range(l):
        parameters["W" + str(i + 1)] -= learning_rate * grads["dW" + str(i + 1)]
        parameters["b" + str(i + 1)] -= learning_rate * grads["db" + str(i + 1)]

    return parameters


def initialize_momentum(parameters):
    # momentum initialization
    l = len(parameters) // 2
    v = {}
    for i in range(l):
        v["Vdw" + str(i + 1)] = np.zeros((parameters["W" + str(i + 1)].shape[0], parameters["W" + str(i + 1)].shape[1]))
        v["Vdb" + str(i + 1)] = np.zeros((parameters["b" + str(i + 1)].shape[0], 1))

    return v


def update_with_momentum(parameters, v, grads, beta1, learning_rate):
    l = len(parameters) // 2

    for i in range(l):
        v["Vdw" + str(i + 1)] = v["Vdw" + str(i + 1)] * beta1 + (1 - beta1) * grads["dW" + str(i + 1)]
        v["Vdb" + str(i + 1)] = v["Vdb" + str(i + 1)] * beta1 + (1 - beta1) * grads["db" + str(i + 1)]

        parameters["W" + str(i + 1)] -= learning_rate * v["Vdw" + str(i + 1)]
        parameters["b" + str(i + 1)] -= learning_rate * v["Vdb" + str(i + 1)]

    return parameters, v


def initialize_rmsprop(parameters):
    l = len(parameters) // 2
    s = {}
    for i in range(l):
        s["Sdw" + str(i + 1)] = np.zeros((parameters["W" + str(i + 1)].shape[0], parameters["W" + str(i + 1)].shape[1]))
        s["Sdb" + str(i + 1)] = np.zeros((parameters["b" + str(i + 1)].shape[0], 1))

    return s


def update_with_rmsprop(parameters, s, grads, beta2, epsilon, learning_rate):
    l = len(parameters) // 2

    for i in range(l):
        s["Sdw" + str(i + 1)] = s["Sdw" + str(i + 1)] * beta2 + (1 - beta2) * np.multiply(grads["dW" + str(i+1)], grads["dW" + str(i+1)])
        s["Sdb" + str(i + 1)] = s["Sdb" + str(i + 1)] * beta2 + (1 - beta2) * np.multiply(grads["db" + str(i+1)], grads["db" + str(i+1)])

        parameters["W" + str(i + 1)] -= learning_rate * grads["dW" + str(i + 1)] / (np.sqrt(s["Sdw" + str(i + 1)]) + epsilon)
        parameters["b" + str(i + 1)] -= learning_rate * grads["db" + str(i + 1)] / (np.sqrt(s["Sdb" + str(i + 1)]) + epsilon)

    return parameters, s 


class NeuralNetwork(object):

    parameters = {}
    cache = {}
    size = 0


    def __init__(self, layer):
        self.size = len(layer) - 1
        for i in range(1, len(layer)):
            self.parameters["W" + str(i)] = np.random.randn(layer[i], layer[i-1]) * 0.01
            self.parameters["b" + str(i)] = np.zeros((layer[i], 1))


    def feed_forward(self, X):
        self.cache = {}

        # append the X as A0
        self.cache["A0"] = X

        for i in range(self.size):
            z = np.dot(self.parameters["W" + str(i + 1)], self.cache["A" + str(i)]) + self.parameters["b" + str(i + 1)]

            # relu or sigmoid
            if i == self.size - 1:
                a = sigmoid(z)
            else:
                a = relu(z)

            self.cache["Z" + str(i + 1)] = z
            self.cache["A" + str(i + 1)] = a

        return self.cache["A" + str(self.size)]


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
                grads["dZ" + str(i + 1)] = np.dot(self.parameters["W" + str(i + 2)].T, grads["dZ" + str(i + 2)]) * relu(self.cache["Z" + str(i + 1)], deriv=True)

            grads["dW" + str(i + 1)] = np.dot(grads["dZ" + str(i + 1)], self.cache["A" + str(i)].T) / m
            grads["db" + str(i + 1)] = np.sum(grads["dZ" + str(i + 1)], axis=1, keepdims=True) / m

        return grads


    def fit(self, X, y, learning_rate, num_iterations, batch_size=128, validation_set=None, beta1=0.9, 
            beta2=0.999, epsilon=1e-8, optimizer="gd"):
        errors = []
        val_errors = []

        # initialize momentum
        if optimizer == "momentum":
            v = initialize_momentum(self.parameters)
        elif optimizer == "rmsprop":
            s = initialize_rmsprop(self.parameters)

        for _ in range(num_iterations):
            # shuffle the data
            shuffle = np.arange(X.shape[1])
            np.random.shuffle(shuffle)
            shuffle_X = X[:, shuffle]
            shuffle_y = y[:, shuffle]

            # partition the data into mini-batches
            batch_X = [shuffle_X[:, i:i+batch_size] for i in range(0, shuffle_X.shape[1], batch_size)]
            batch_y = [shuffle_y[:, i:i+batch_size] for i in range(0, shuffle_y.shape[1], batch_size)]

            # batch training
            for _X, _y in zip(batch_X, batch_y):
                # validation error calculation
                if validation_set:
                    val_X = validation_set[0]
                    val_y = validation_set[1]
                    val_pred = self.feed_forward(val_X)
                    val_error = self.calculate_cost(val_pred, val_y)
                    val_errors.append(val_error)

                # feed forward
                prediction = self.feed_forward(_X)

                # error calculation
                error = self.calculate_cost(prediction, _y)
                errors.append(error)

                # backpropagation
                grads = self.backward_propagation(prediction, _y)

                # update parameters
                if optimizer == "gd":
                    self.parameters = update_with_gd(self.parameters, grads, learning_rate)
                elif optimizer == "momentum":
                    self.parameters, v = update_with_momentum(self.parameters, v, grads, beta1, learning_rate)
                elif optimizer == "rmsprop":
                    self.parameters, s = update_with_rmsprop(self.parameters, s, grads, beta2, epsilon, learning_rate)

            print("Iteration: {}  |  Error: {}".format(_, error))

        return errors, val_errors


    def evaluate(self, test_data, test_labels):
        prediction = self.feed_forward(test_data)
        prediction = np.argmax(prediction, axis=0)
        true_label = np.argmax(test_labels, axis=0)

        accuracy = np.sum(prediction == true_label) / test_labels.shape[1]

        return accuracy


if __name__ == "__main__":
    np.random.seed(23)

    X_train, X_val, y_train, y_val = pre_process_dataset("train.csv")

    model = NeuralNetwork([784, 100, 100, 10])

    errors, val_errors = model.fit(X_train, y_train, learning_rate=0.4, num_iterations=5, optimizer="momentum")

    # training + validation, take longer times
    # errors, val_errors = model.fit(X_train, y_train, learning_rate=0.4, num_iterations=5, validation_set=(X_val, y_val))

    # evaluate accuracy
    training_accuracy = model.evaluate(X_train, y_train)
    print("Training Accuracy: ", training_accuracy)

    accuracy = model.evaluate(X_val, y_val)
    print("Validation Accuracy: ", accuracy)

    # save model
    filename = "NN_" + str(accuracy)
    pickle.dump(model, open(filename, "wb"))

    # error graph
    plt.title("Errors")
    plt.plot(errors, label="Training")
    if val_errors:
        plt.plot(val_errors, label="Validation")
    plt.legend()
    plt.xlabel("Batches")
    plt.ylabel("Errors")
    plt.show()