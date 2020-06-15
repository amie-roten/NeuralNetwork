# Amie Roten
# CS559: Term Project
# Neural Network Implementation

import random
from pathlib import Path

import numpy as np
import utils
from matplotlib import pyplot as plt

np.set_printoptions(suppress=True)

####
###
##
#  Layer class for neural network.
class Layer:
    def __init__(self, nodes, activation, activation_deriv):
        self.weights = []
        self.nodes = nodes
        self.activation = activation
        self.activation_deriv = activation_deriv
        self.z = np.zeros(self.nodes)
        self.activations = np.zeros(self.nodes)
        self.delta = np.zeros(self.nodes)

    # Initialize weights.
    def initialize_weights(self, input_num):
        # Weights initialized using Xavier initialization,
        # this is better for fixed-range activations (sigmoid,
        # tanh):
        if self.activation == utils.sigmoid or \
           self.activation == utils.tanh:
            self.weights = (np.random.rand(self.nodes, input_num) - 0.5) * 2
            self.weights = self.weights * (np.sqrt(6)/(np.sqrt(input_num + self.nodes)))
        # Weights initialized using Kaiming initialization,
        # this is better for non-fixed-range activations (ReLU):
        elif self.activation == utils.relu:
            self.weights = np.random.standard_normal((self.nodes, input_num)) * \
                           (np.sqrt(2)/np.sqrt(input_num))
            self.weights[:,0] = 0
        # Weights initialized between -0.5 - 0.5:
        else:
            self.weights = np.random.rand(self.nodes, input_num) - 0.5

    # Perform forward pass step for this layer.
    def forward_pass(self, input):
        self.z = self.weights @ input
        self.activations = self.activation(self.z)
        return self.activations

    # Perform backpropogation step for this
    # layer. This process/formula applied to
    # hidden layers only.
    def backpropagation(self, next_delta, next_weights):
        self.delta = np.multiply((next_weights.T @ next_delta), self.activation_deriv(self.z))

    # Update weights using errors from backprop.
    def weight_update(self, learning_rate, prev_activations):
        self.weights = self.weights - learning_rate * np.outer(self.delta, prev_activations)

class NeuralNetwork:
    def __init__(self,
                 input_nodes,
                 output_nodes,
                 output_activation,
                 output_deriv,
                 objective_fcn,
                 learning_rate = 0.2):
        self.layers = []
        self.objective_fcn = objective_fcn
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        if output_nodes == 1:
            self.single_target = True
        else:
            self.single_target = False
        self.learning_rate = learning_rate
        self.output_layer = Layer(self.output_nodes,
                                  output_activation,
                                  output_deriv)

    # Add a hidden layer to the network.
    def add_layer(self, layer):
        self.layers.append(layer)

    # Simple visualization to confirm
    # network architecture.
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
    def fit(self, X, y,
            val_X = None, val_y = None,
            epochs = 10, include_bias = True,
            early_stopping=False,
            patience=2,
            plot_name=""):
        self.visualize()
        last_val_error = float("inf")
        val_increase = 0
        all_train_errors = []
        all_val_errors = []

        # Adding bias/intercept term
        # to X input as an initial
        # column of ones.
        if include_bias:
            self.input_nodes = self.input_nodes + 1
            X = np.vstack((np.ones(X.shape[0]), X.T)).T

        # Randomly initializing layer weights.
        input_num = X.shape[1]
        for layer in self.layers:
            layer.initialize_weights(input_num)
            input_num = layer.nodes
        self.output_layer.initialize_weights(input_num)

        # Running on-line gradient descent for
        # specified number of epochs.
        for epoch in range(epochs):
            all_error = []
            all_outputs = []
            choices = list(range(X.shape[0]))
            random.shuffle(choices)
            while choices != []:
                index = choices.pop(0)
                x_i = X[index,:]

                if self.output_nodes == 1:
                    y_i = y[index]
                else:
                    y_i = y[:, index]

                # Calculate f(x_i, w) using forward pass
                # through all layers, and determine error.
                inputs = x_i
                for layer in self.layers:
                    inputs = layer.forward_pass(inputs)
                final_outputs = self.output_layer.forward_pass(inputs)

                if np.isnan(sum(final_outputs)):
                    print("Exploding output values, try a smaller learning rate\n Exiting training...")
                    return

                all_outputs.append(final_outputs)
                all_error.append(self.objective_fcn(y_i, final_outputs))

                # Calculate derivatives using backpropagation.
                self.output_layer.delta = (final_outputs - y_i)
                delta = self.output_layer.delta
                weights = self.output_layer.weights
                for layer in self.layers[::-1]:
                    layer.backpropagation(delta, weights)
                    delta = layer.delta
                    weights = layer.weights

                # Update all weights.
                activations = x_i
                for layer in self.layers:
                    layer.weight_update(self.learning_rate, activations)
                    activations = layer.activations
                self.output_layer.weight_update(self.learning_rate, activations)

            if val_X is not None:
                _, val_error = self.evaluate(val_X, val_y)

            all_train_errors.append(sum(all_error) / len(all_error))
            all_val_errors.append(val_error)

            # print("Completed epoch", epoch+1)
            # print("\tAverage epoch test error:", sum(all_error) / len(all_error))
            # print("\tAverage epoch validation error:", val_error)

            if val_error > last_val_error:
                val_increase += 1
            else:
                val_increase = 0
            if early_stopping:
                if val_increase == patience:
                    print("Early stopping due to", val_increase, "epochs with increasing validation error...")
                    print("Total epochs run:", epoch+1)
                    epoch += 1 # for plotting reasons
                    break
            last_val_error = val_error

        if "visualize":
            x = list(range(1, len(all_val_errors)+1))
            plt.plot(x, all_val_errors, label="Validation Error")
            plt.plot(x, all_train_errors, label="Training Error")
            plt.ylabel("Mean Error", fontsize=16)
            plt.xlabel("Epoch", fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.title(plot_name.split("\n",maxsplit=1)[1],fontsize=18)
            plt.legend(fontsize=16)
            filename = plot_name.replace("\n", " ") + ".png"
            filename = filename.replace(" ", "_").replace(",","")
            plt.savefig(Path("figs", filename))
            plt.show()

    # Returns mean classification % accuracy, if applicable, as well
    # as average error across all samples, using the model's
    # particular objective function.
    def evaluate(self, X, y):
        correct = 0
        incorrect = 0
        all_error = []
        X = np.vstack((np.ones(X.shape[0]), X.T)).T
        for i in range(X.shape[0]):
            x_i = X[i, :]

            if self.single_target:
                y_i = y[i]
            else:
                y_i = y[:, i]

            # Calculate y_hat using forward pass
            # through all layers, then argmax.
            inputs = x_i
            for layer in self.layers:
                inputs = layer.forward_pass(inputs)
            final_outputs = self.output_layer.forward_pass(inputs)

            if self.objective_fcn == utils.binary_crossentropy:
                y_hat = np.round(final_outputs)
                y_class = y_i
                if y_hat == y_class:
                    correct += 1
                else:
                    incorrect += 1

            if self.objective_fcn == utils.multiclass_crossentropy or \
                    (self.objective_fcn == utils.SSE and self.output_nodes > 1):
                y_hat = np.argmax(final_outputs)
                y_class = np.argmax(y_i)
                if y_hat == y_class:
                    correct += 1
                else:
                    incorrect += 1

            all_error.append(self.objective_fcn(y_i, final_outputs))

        if self.objective_fcn == utils.binary_crossentropy or \
                self.output_nodes > 1:
            return correct/(correct+incorrect)*100, sum(all_error) / len(all_error)
        return None, sum(all_error) / len(all_error)

    def predict(self, X):
        y_hat_all = []
        X = np.vstack((np.ones(X.shape[0]), X.T)).T
        for i in range(X.shape[0]):
            x_i = X[i, :]

            # Calculate y_hat using forward pass
            # through all layers.
            inputs = x_i
            for layer in self.layers:
                inputs = layer.forward_pass(inputs)
            final_outputs = self.output_layer.forward_pass(inputs)

            if self.objective_fcn == utils.binary_crossentropy:
                y_hat_all.append(np.round(final_outputs))
            elif self.objective_fcn == utils.multiclass_crossentropy or \
                    self.output_nodes > 1:
                y_hat_all.append(np.argmax(final_outputs))
            else:
                y_hat_all.append(final_outputs)

        return np.array(y_hat_all)

if __name__ == "__main__":
    print("nothing to see here!")