import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


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


def initialize_random(n_x, n_h, n_y):
	parameters = {}
	parameters["W1"] = np.random.randn(n_h, n_x) * 0.01
	parameters["b1"] = np.zeros((n_h, 1))
	parameters["W2"] = np.random.randn(n_y, n_h) * 0.01
	parameters["b2"] = np.zeros((n_y, 1))
	return parameters


def relu(x, deriv=False):
	if deriv:
		return 1 * (x > 0)
	return np.maximum(0, x)


def sigmoid(x, deriv=False):
	if deriv:
		return sigmoid(x) * (1 - sigmoid(x))
	return 1 / ( 1 + np.exp(-x))


def feed_forward(X, parameters):
	cache = {}

	# unpack the parameters
	W1 = parameters["W1"]
	W2 = parameters["W2"]
	b1 = parameters["b1"]
	b2 = parameters["b2"]

	# feed forward
	Z1 = np.dot(W1, X) + b1
	A1 = relu(Z1)
	Z2 = np.dot(W2, A1) + b2
	A2 = sigmoid(Z2)

	# cache the Z and A for backward propagation
	cache["A0"] = X
	cache["Z1"] = Z1
	cache["A1"] = A1
	cache["Z2"] = Z2
	cache["A2"] = A2
	cache["W2"] = W2

	return cache


def calculate_cost(AL, y):
	m = y.shape[1]
	loss = np.sum(y * np.log(AL) + (1-y) * np.log(1-AL))
	return loss / (-m)


def backward_propagation(y, cache):
	m = y.shape[1]

	# unpack the cache
	A2 = cache["A2"]
	Z2 = cache["Z2"]
	A1 = cache["A1"]
	Z1 = cache["Z1"]
	A0 = cache["A0"]
	W2 = cache["W2"]

	dZ2 = A2 - y
	dW2 = np.dot(dZ2, A1.T) / m
	db2 = np.sum(dZ2, axis=1, keepdims=True) / m

	dZ1 = np.dot(W2.T, dZ2) * relu(Z1, deriv=True)
	dW1 = np.dot(dZ1, A0.T) / m
	db1 = np.sum(dZ1, axis=1, keepdims=True) / m

	# save the gradients
	grads = {}
	grads["dW1"] = dW1
	grads["db1"] = db1
	grads["dW2"] = dW2
	grads["db2"] = db2

	return grads


def update_parameters(parameters, grads, learning_rate):
	parameters["W1"] -= learning_rate * grads["dW1"]
	parameters["b1"] -= learning_rate * grads["db1"]
	parameters["W2"] -= learning_rate * grads["dW2"]
	parameters["b2"] -= learning_rate * grads["db2"]

	return parameters


def evaluate(parameters, test_data, test_labels):
	cache = feed_forward(test_data, parameters)
	prediction = np.argmax(cache["A2"], axis=0)
	true = np.argmax(test_labels, axis=0)

	accuracy = np.sum(prediction == true) / test_labels.shape[1]
	return accuracy


if __name__ == '__main__':
	# seed
	np.random.seed(23)

	# load the data
	X_train, X_test, y_train, y_test = pre_process_dataset("train.csv")

	# Neural Network Hyperparameters
	learning_rate = 0.4
	n_x = 784
	n_h = 100
	n_y = 10
	num_iterations = 100

	# Neural Network Parameters
	parameters = initialize_random(n_x, n_h, n_y)

	# training
	errors = []
	for _ in range(num_iterations):
		cache = feed_forward(X_train, parameters)
		error = calculate_cost(cache["A2"], y_train)
		errors.append(error)
		print("Iteration: {}  |  Cost: {}". format(_, error))
		grads = backward_propagation(y_train, cache)
		parameters = update_parameters(parameters, grads, learning_rate)

	# evaluate
	accuracy = evaluate(parameters, X_test, y_test)
	print("Accuracy: ", accuracy)