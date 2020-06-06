# Amie Roten
# CS559: Term Project
# Neural Network Implementation

from typing import List
import random

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
#import data

np.set_printoptions(suppress=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.max(x, np.zeros(x.shape[0]))

def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def identity(x):
    return x

class Layer:
    def __init__(self, nodes, activation):
        self.weights = []
        self.nodes = nodes
        self.activation = activation
        self.activations = np.zeros(self.nodes)
        self.z = np.zeros(self.nodes)
        self.delta = np.zeros(self.nodes)

    def initialize_weights(self, input_num):
        self.weights = np.random.rand(self.nodes, input_num) - 0.5

    def forward_pass(self, input):
        self.z = self.weights @ input
        self.activations = self.activation(self.z)
        return self.activations

    # This is really just computing partial
    # derivatives across the entire cost
    # function! http://neuralnetworksanddeeplearning.com/chap2.html
    def backpropagation(self, next_delta, next_weights):
        self.delta = sum(next_delta @ next_weights) * \
                     self.activations * (1 - self.activations)

    def weight_update(self, learning_rate, prev_activations):
        self.weights = self.weights - learning_rate * np.outer(self.delta,prev_activations)

class NeuralNetwork:
    def __init__(self,
                 input_nodes,
                 output_nodes,
                 output_activation,
                 objective_fcn = str,
                 learning_rate = 0.2):
        self.layers = []
        self.objective_fcn = objective_fcn
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate
        self.output_layer = Layer(self.output_nodes,
                                  output_activation)

    def add_layer(self, layer):
        self.layers.append(layer)

    def visualize(self):
        nodes = ""
        for node in range(self.input_nodes):
            nodes += "o    "
        print("\t\tInput layer:", nodes)
        for layer in range(len(self.layers)):
            nodes = ""
            for node in range(self.layers[layer].nodes):
                nodes += "o    "
            print("\tHidden layer", layer+1, ":", nodes)
        nodes = ""
        for node in range(self.output_nodes):
            nodes += "o    "
        print("\t\tOutput layer:", nodes)


    # Online gradient descent training.
    # Assuming for multi-class classification,
    # y is already in one-hot form.
    def fit(self, X, y, epochs = 10):

        # Adding bias/intercept term
        # to X input as an initial
        # column of ones.
        X = np.vstack((np.ones(X.shape[0]), X.T)).T
        self.input_nodes = X.shape[1]
        self.output_nodes = y.shape[0]

        # Randomly initializing layer weights.
        input_num = X.shape[1]
        for layer in self.layers:
            layer.initialize_weights(input_num)
            input_num = layer.nodes
        self.output_layer.initialize_weights(input_num)

        # Running on-line gradient descent for
        # specified number of epochs.
        for epoch in range(epochs):
            choices = list(range(X.shape[0]))
            random.shuffle(choices)
            while choices != []:
                index = choices.pop(0)
                x_i = X[index,:]
                y_i = y[:, index]

                # Calculate f(x_i, w) using forward pass
                # through all layers.
                inputs = x_i
                for layer in self.layers:
                    inputs = layer.forward_pass(inputs)
                final_outputs = self.output_layer.forward_pass(inputs)

                # Calculate derivatives using backpropagation.
                if self.objective_fcn == "MSE":
                    self.output_layer.delta = -(y_i - final_outputs)
                delta = self.output_layer.delta
                weights = self.output_layer.weights
                for layer in self.layers:
                    layer.backpropagation(delta, weights)
                    delta = layer.delta
                    weights = layer.weights

                # Update all weights.
                activations = x_i
                for layer in self.layers:
                    layer.weight_update(self.learning_rate, activations)
                    activations = layer.activations
                self.output_layer.weight_update(self.learning_rate, activations)

    def evaluate(self, X, y):
        correct = 0
        incorrect = 0
        X = np.vstack((np.ones(X.shape[0]), X.T)).T
        for i in range(X.shape[0]):
            x_i = X[i, :]
            y_i = y[:, i]

            # Calculate y_hat using forward pass
            # through all layers, then argmax.
            inputs = x_i
            for layer in self.layers:
                inputs = layer.forward_pass(inputs)
            final_outputs = self.output_layer.forward_pass(inputs)
            y_hat = np.argmax(final_outputs)
            y_class = np.argmax(y_i)
            if y_hat == y_class:
                correct += 1
            else:
                incorrect += 1

        print("Network weights achieved", correct/(correct+incorrect)*100, "% accuracy.")

def one_hot(y):
    y_onehot = []
    for target in y:
        onehot = np.zeros(3)
        onehot[target] = 1
        y_onehot.append(onehot)
    return np.array(y_onehot).T

def preprocess_iris(binary: bool = False, features: int = 4):
    iris = datasets.load_iris()
    y = iris.target
    X = iris.data
    y_onehot = one_hot(y)
    return X, y_onehot

if __name__ == "__main__":
    X, y = preprocess_iris()
    X_train, X_test, y_train, y_test = train_test_split(X, y.T, test_size=0.2)
    model = NeuralNetwork(input_nodes = X.shape[1],
                          output_nodes = y.shape[0],
                          output_activation=softmax,
                          objective_fcn="MSE")
    model.add_layer(Layer(nodes=10, activation=sigmoid))
    model.add_layer(Layer(nodes=10, activation=sigmoid))
    model.fit(X_train, y_train.T, epochs=1000)
    #model.fit(X, y)
    model.evaluate(X_test, y_test.T)
    print("pause")