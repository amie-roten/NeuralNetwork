# Amie Roten
# CS559: Term Project
# Neural Network Implementation

from typing import List

import numpy as np
from sklearn import datasets
#import data

np.set_printoptions(suppress=True)

# Defining sigmoid function.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class HiddenLayer:
    def __init__(self, k, j):
        self.weights = np.array((j, k))

class NeuralNetwork:
    def __init__(self, hidden_layers = int,
                       num_nodes = List,
                       objective_fcn = str):
        if hidden_layers != len(num_nodes):
            raise Exception("Node input does not match number of hidden",
                            "layers!")
        self.hidden_layers = hidden_layers
        self.num_nodes = num_nodes
        self.objective_fcn = objective_fcn
        self.weights = []
        self.input_nodes = None
        self.output_nodes = None
        self.learning_rate = 0.2

    # Online gradient descent training.
    # Assuming for multi-class classification,
    # y is already in one-hot form.
    def fit(self, X, y):
        self.input_nodes = X.shape[1]
        self.output_nodes = y.shape[0]
        layer_shapes = [self.input_nodes] + self.num_nodes + [self.output_nodes]
        self.weights = [np.random.rand(layer_shapes[layer+1], layer_shapes[layer]) - 0.5 \
                        for layer in range(len(layer_shapes) - 1)]
        for i in range(X.shape[0]):
            z, activations = self.__forward_pass(X[i, :], y[:, i], layer_shapes)
            derivatives = self.__backpropogation(z, activations)
            # Update weights using derivatives.
            # weights = weights - learning rate * derivative * activation of previous
            print("pause")


    def __forward_pass(self, X, y, layer_shapes):
        z = [np.zeros(layer_shapes[layer]) \
                       for layer in range(self.hidden_layers + 2)]
        activations = [np.zeros(layer_shapes[layer]) \
                       for layer in range(self.hidden_layers + 2)]
        z[0] = X
        activations[0] = X
        for layer in range(0,len(activations)-1):
            activations[layer + 1] = sigmoid(self.weights[layer] @ activations[layer])
            z[layer + 1] = self.weights[layer] @ activations[layer]
        # Output layer!
        return z, activations

    # This is really just computing partial
    # derivatives across the entire cost
    # function! http://neuralnetworksanddeeplearning.com/chap2.html
    def __backpropogation(self, z, activations):
        raise NotImplementedError

    def evaluate(self, X, y):
        raise NotImplementedError

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
    model = NeuralNetwork(hidden_layers = 1,
                          num_nodes = [10],
                          objective_fcn="MSE")
    model.fit(X, y)
    print("pause")