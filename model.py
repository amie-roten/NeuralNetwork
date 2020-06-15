# Amie Roten
# CS559: Term Project
# Neural Network Implementation

from typing import List
import random
from pathlib import Path
import pickle as pkl

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import data
import utils
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import seaborn as sn

np.set_printoptions(suppress=True)


class Layer:
    def __init__(self, nodes, activation, activation_deriv):
        self.weights = []
        self.nodes = nodes
        self.activation = activation
        self.activation_deriv = activation_deriv
        self.z = np.zeros(self.nodes)
        self.activations = np.zeros(self.nodes)
        self.delta = np.zeros(self.nodes)


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

    def forward_pass(self, input):
        self.z = self.weights @ input
        self.activations = self.activation(self.z)
        return self.activations

    # This is really just computing partial
    # derivatives across the entire cost
    # function! http://neuralnetworksanddeeplearning.com/chap2.html
    def backpropagation(self, next_delta, next_weights):
        self.delta = np.multiply((next_weights.T @ next_delta), self.activation_deriv(self.z))

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

def get_PCA(X_train, y_train, X_test, y_test, X_val, y_val,
            visualize=False, features=3):
    pca = PCA(n_components=features)
    pca.fit(X_train)
    X_train_reduced = pca.transform(X_train)
    X_test_reduced = pca.transform(X_test)
    X_val_reduced = pca.transform(X_val)
    variance = pca.explained_variance_ratio_
    if visualize:
        print(variance)
        components = np.concatenate((X_train_reduced, X_test_reduced))
        components = np.concatenate((components, X_val_reduced))
        try:
            y = np.concatenate((y_train, y_test), axis=0)
            y = np.concatenate((y, y_val), axis=0)
            plt.scatter(components[:,0], components[:,1], cmap="Paired",
                        c = np.argmax(y, axis=1))
        except:
            y = np.concatenate((y_train, y_test), axis=0)
            y = np.concatenate((y, y_val), axis=0)
            plt.scatter(components[:, 0], components[:, 1], cmap="Paired",
                        c=y)
        plt.title("PCA Visualization\nEmotions Data")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.show()
    return X_train_reduced, X_test_reduced, X_val_reduced

# Experiment with network using Boston Housing
# dataset, to evaluate network functionality
# to solve a regression problem. We expect a
# low error, at least comparable to the results
# from HW2, where we trained linear regression
# models on the same data.

def regression_experiment():

    boston = datasets.load_boston()
    y = boston.target
    X = boston.data

    X_train, X_test, y_train, y_test = train_test_split(X, y.T, test_size=0.2, random_state=10)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.2, random_state=10)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)

    nodes = 16

    for activation in ["Sigmoid", "Tanh", "ReLU"]:
    #for activation in ["Linear"]:
        for layers in range(4, 5):
            random.seed(100)
            np.random.seed(100)
            model = NeuralNetwork(input_nodes=X.shape[1],
                                  output_nodes=1,
                                  output_activation=utils.identity,
                                  output_deriv=utils.identity_deriv,
                                  objective_fcn=utils.SSE,
                                  learning_rate=0.001)
            i = 0
            if activation == "Sigmoid":
                model.learning_rate = 0.01
                while i < layers:
                    model.add_layer(Layer(nodes=nodes,
                                          activation=utils.sigmoid,
                                          activation_deriv=utils.sigmoid_deriv))
                    i += 1
            elif activation == "Tanh":
                model.learning_rate = 0.001
                while i < layers:
                    model.add_layer(Layer(nodes=nodes,
                                          activation=utils.tanh,
                                          activation_deriv=utils.tanh_deriv))
                    i += 1
            elif activation == "ReLU":
                model.learning_rate = 0.001
                while i < layers:
                    model.add_layer(Layer(nodes=nodes,
                                          activation=utils.relu,
                                          activation_deriv=utils.relu_deriv))
                    i += 1

            model.fit(X_train,
                      y_train.T,
                      X_val,
                      y_val.T,
                      epochs=20,
                      include_bias=True,
                      early_stopping=False,
                      plot_name="Boston Housing Regression\n"+str(layers)+" Hidden Layers, with " + str(nodes) + " Nodes and\n"+ activation+" Activation")

            class_error, error = model.evaluate(X_test,
                                                y_test.T)

            print(str(layers)+" Hidden Layers, with "+ activation+" Activation")
            print("Test mean error:", error, "\n\n")

# Another experiment using a simple dataset, in
# order to validate framework functionality for
# binary classification. This experiment uses
# the Breast Cancer Wisconsin dataset, which
# is an easy dataset, and we expect the model
# to perform well.
def binary_classification_experiment():

    cancer = datasets.load_breast_cancer()
    y = cancer.target
    X = cancer.data

    X_train, X_test, y_train, y_test = train_test_split(X, y.T, test_size=0.2, random_state=10)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.2, random_state=10)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)

    if "visualize PCA features":
        num_features = 3
        get_PCA(X_train, y_train,
                X_test, y_test,
                X_val, y_val,
                visualize=True,
                features=2)

    nodes = 16


    for activation in ["Sigmoid", "Tanh", "ReLU"]:
    #for activation in ["Linear"]:
        for layers in range(1, 2):
            random.seed(100)
            np.random.seed(100)
            model = NeuralNetwork(input_nodes=X.shape[1],
                                  output_nodes=1,
                                  output_activation=utils.sigmoid,
                                  output_deriv=utils.sigmoid_deriv,
                                  objective_fcn=utils.binary_crossentropy,
                                  learning_rate=0.001)
            i = 0
            if activation == "Sigmoid":
                model.learning_rate = 0.01
                while i < layers:
                    model.add_layer(Layer(nodes=nodes,
                                          activation=utils.sigmoid,
                                          activation_deriv=utils.sigmoid_deriv))
                    i += 1
            elif activation == "Tanh":
                model.learning_rate = 0.001
                while i < layers:
                    model.add_layer(Layer(nodes=nodes,
                                          activation=utils.tanh,
                                          activation_deriv=utils.tanh_deriv))
                    i += 1
            elif activation == "ReLU":
                model.learning_rate = 0.001
                while i < layers:
                    model.add_layer(Layer(nodes=nodes,
                                          activation=utils.relu,
                                          activation_deriv=utils.relu_deriv))
                    i += 1

            model.fit(X_train,
                      y_train.T,
                      X_val,
                      y_val.T,
                      epochs=20,
                      include_bias=True,
                      early_stopping=False,
                      plot_name="Cancer Binary Classification\n"+str(layers)+" Hidden Layers, with " + str(nodes) + " Nodes and\n"+ activation+" Activation")

            class_error, error = model.evaluate(X_test,
                                                y_test.T)

            print(str(layers)+" Hidden Layers, with "+ activation+" Activation")
            print("Test mean error:", error)
            print("Test classification accuracy:", class_error, "\n\n")


# A final simple experiment to validate the
# framework on a multiclass classification
# problem. This is the classic Iris dataset,
# and we expect the model to perform well!
def multiclass_classification_experiment():

    iris = datasets.load_iris()
    X = iris.data
    # A concise, quick way to create one-hot targets! Thanks to:
    # https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-1-hot-encoded-numpy-array
    y = np.eye(max(iris.target)+1)[iris.target]
    classes = max(iris.target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10, stratify=y)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.2, random_state=10, stratify=y_test)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)

    nodes = 8

    if not "visualize PCA features":
        num_features = 3
        get_PCA(X_train, y_train,
                X_test, y_test,
                X_val, y_val,
                visualize=True,
                features=2)

    for activation in ["Sigmoid", "Tanh", "ReLU"]:
    #for activation in ["Linear"]:
        for layers in range(1, 3):
            random.seed(100)
            np.random.seed(100)
            model = NeuralNetwork(input_nodes=X.shape[1],
                                  output_nodes=classes+1,
                                  output_activation=utils.softmax,
                                  output_deriv=utils.softmax_deriv,
                                  objective_fcn=utils.multiclass_crossentropy,
                                  learning_rate=0.001)
            i = 0
            if activation == "Sigmoid":
                model.learning_rate = 0.01
                while i < layers:
                    model.add_layer(Layer(nodes=nodes,
                                          activation=utils.sigmoid,
                                          activation_deriv=utils.sigmoid_deriv))
                    i += 1
            elif activation == "Tanh":
                model.learning_rate = 0.01
                while i < layers:
                    model.add_layer(Layer(nodes=nodes,
                                          activation=utils.tanh,
                                          activation_deriv=utils.tanh_deriv))
                    i += 1
            elif activation == "ReLU":
                model.learning_rate = 0.01
                while i < layers:
                    model.add_layer(Layer(nodes=nodes,
                                          activation=utils.relu,
                                          activation_deriv=utils.relu_deriv))
                    i += 1

            model.fit(X_train,
                      y_train.T,
                      X_val,
                      y_val.T,
                      epochs=50,
                      include_bias=True,
                      early_stopping=False,
                      plot_name="Iris Multiclass Classification\n" + str(layers) + " Hidden Layers, with " + str(
                          nodes) + " Nodes and\n" + activation + " Activation")

            class_error, error = model.evaluate(X_test,
                                                y_test.T)
            y_hat = model.predict(X_test)

            if activation == "Linear":
                conf = confusion_matrix(np.argmax(y_test, axis=1), y_hat)
                print(conf)
                sn.heatmap(conf, cmap="Blues", annot=True,
                           xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
                plt.xlabel("Predicted Label")
                plt.ylabel("True Label")
                plt.show()

            print(str(layers) + " Hidden Layers, with " + activation + " Activation")
            print("Test mean error:", error)
            print("Test classification accuracy:", class_error, "\n\n")


def emotions_experiment():
    corpus = data.Corpus()
    if "split_genders":
        X, y_class = corpus.get_all_data_gendered()
    else:
        X, y_class = corpus.get_all_data()

    best_params = {}
    best_params["class_acc"] = 0
    best_models = []

    # This is finicky and brittle. The classes
    # must be listed in-order to be able to interpret
    # resulting classification.
    if not "subset classes":
        classes = ["happy", "sad", "angry"]
        classes = [data.emotion_map_find_class[emotion] for emotion in classes]
        zipped = list(zip(X, y_class))
        X = np.array([x for x, y in zipped if y in classes])
        y = [y for x, y in zipped if y in classes]
        y = np.eye(max(y) + 1)[y]
        # Again, refined the process of removing all zero
        # columns with help from the folks at StackOverflow!
        # https://stackoverflow.com/questions/51769962/
        # find-and-delete-all-zero-columns-from-numpy-array-using-fancy-indexing/51770365
        i = np.argwhere(np.all(y[..., :] == 0, axis=0))
        y = np.delete(y, i, axis=1)
        classes = len(classes)
    else:
        classes = max(y_class) + 1
        y = np.eye(max(y_class) + 1)[y_class]

    if not "remove neutral":
        X = np.array([X[i,:] for i in range(y.shape[0]) if y_class[i] != 0])
        y = np.array([y[i,:] for i in range(y.shape[0]) if y_class[i] != 0])
        classes = classes - 1
        i = np.argwhere(np.all(y[..., :] == 0, axis=0))
        y = np.delete(y, i, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=10, stratify=y)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.20, random_state=10, stratify=y_test)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)
    X_all = scaler.transform(X)

    nodes = 24

    if not "use PCA features":
        num_features = 3
        X_train, X_test, X_val = get_PCA(X_train, y_train,
                                         X_test, y_test,
                                         X_val, y_val,
                                         visualize=True,
                                         features=3)
    if not "experiment 1":
        experiments = 0
        for learning_rate in [0.1, 0.01, 0.005, 0.001]:
            for nodes in [12, 24, 36, 48]:
                for activation in ["Sigmoid", "Tanh", "ReLU"]:
                    for layers in [1, 2, 3, 4, 5, 10, 20, 25]:
                        experiments += 1
                        random.seed(100)
                        np.random.seed(100)
                        model = NeuralNetwork(input_nodes=X_train.shape[1],
                                              output_nodes=classes,
                                              output_activation=utils.softmax,
                                              output_deriv=utils.softmax_deriv,
                                              objective_fcn=utils.multiclass_crossentropy,
                                              learning_rate=learning_rate)
                        i = 0
                        if activation == "Sigmoid":
                            #model.learning_rate = 0.01
                            while i < layers:
                                model.add_layer(Layer(nodes=nodes,
                                                      activation=utils.sigmoid,
                                                      activation_deriv=utils.sigmoid_deriv))
                                i += 1
                        elif activation == "Tanh":
                            #model.learning_rate = 0.01
                            while i < layers:
                                model.add_layer(Layer(nodes=nodes,
                                                      activation=utils.tanh,
                                                      activation_deriv=utils.tanh_deriv))
                                i += 1
                        elif activation == "ReLU":
                            #model.learning_rate = 0.01
                            while i < layers:
                                model.add_layer(Layer(nodes=nodes,
                                                      activation=utils.relu,
                                                      activation_deriv=utils.relu_deriv))
                                i += 1

                        model.fit(X_train,
                                  y_train.T,
                                  X_val,
                                  y_val.T,
                                  epochs=500,
                                  include_bias=True,
                                  early_stopping=True,
                                  patience=3,
                                  plot_name="Emotions Multiclass Classification\n" + str(layers) + " Hidden Layers, with " + str(
                                      nodes) + " Nodes,\n" + activation + " Activation and " + str(learning_rate) + " Learning Rate")

                        class_acc, error = model.evaluate(X_test,
                                                          y_test.T)

                        if class_acc >= 35:
                            top_model = {}
                            top_model["class_acc"] = class_acc
                            top_model["error"] = error
                            top_model["act_fcn"] = activation
                            top_model["layers"] = layers
                            top_model["nodes"] = nodes
                            top_model["model"] = model
                            top_model["learning rate"] = learning_rate
                            best_models.append(top_model.copy())
                            print("model added")

                        if class_acc > best_params["class_acc"]:
                            best_params["class_acc"] = class_acc
                            best_params["error"] = error
                            best_params["act_fcn"] = activation
                            best_params["layers"] = layers
                            best_params["nodes"] = nodes
                            best_params["model"] = model
                            best_params["learning rate"] = learning_rate

                        print(str(layers) + " Hidden Layers, with " + activation + " Activation")
                        print("Test mean error:", error)
                        print("Test classification accuracy:", class_acc, "\n\n")

        y_hat = best_params["model"].predict(X_test)
        y_test_class = np.argmax(y_test, axis=1)
        conf = confusion_matrix(y_test_class, y_hat)
        if classes == 16:
            labels_male = [data.emotion_map[x+1] + "_male" for x in range(int(classes/2))]
            labels_female = [data.emotion_map[x + 1] + "_female" for x in range(int(classes/2))]
            labels = labels_male + labels_female
        else:
            labels = [data.emotion_map[x+1] for x in range(int(classes))]
        sn.heatmap(conf, cmap="Blues", annot=True,
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()
        print("Best model:")
        print("\tAccuracy:", best_params["class_acc"])
        print("\tMean error:", best_params["error"])
        print("\tActivation:", best_params["act_fcn"])
        print("\tNumber of Layers:", best_params["layers"])
        print("\tNumber of Nodes:", best_params["nodes"])
        print("\tLearning Rate:", best_params["learning rate"])

        print("\nNumber of Experiments:", experiments)

        print("\nAll models with over 30% accuracy:")
        print(best_models)

    if "experiment 2":
        good_models = [{'class_acc': 38.59649122807017, 'error': 1.748976200668625, 'act_fcn': 'Sigmoid', 'layers': 2, 'nodes': 12,
             'learning rate': 0.1},
            {'class_acc': 35.96491228070175, 'error': 1.7847757678261122, 'act_fcn': 'Sigmoid', 'layers': 2, 'nodes': 24,
             'learning rate': 0.1},
            {'class_acc': 35.96491228070175, 'error': 1.807378594087159, 'act_fcn': 'Sigmoid', 'layers': 3, 'nodes': 24,
             'learning rate': 0.1},
            {'class_acc': 36.40350877192983, 'error': 1.9320921600250356, 'act_fcn': 'Tanh', 'layers': 1, 'nodes': 24,
             'learning rate': 0.1},
            {'class_acc': 36.84210526315789, 'error': 1.783922398229393, 'act_fcn': 'Sigmoid', 'layers': 2, 'nodes': 36,
             'learning rate': 0.1},
            {'class_acc': 38.15789473684211, 'error': 1.8449377552886115, 'act_fcn': 'Sigmoid', 'layers': 1, 'nodes': 48,
             'learning rate': 0.1},
            {'class_acc': 37.280701754385966, 'error': 1.8395186552397325, 'act_fcn': 'Sigmoid', 'layers': 2, 'nodes': 48,
             'learning rate': 0.1},
            {'class_acc': 35.526315789473685, 'error': 1.7573407279415063, 'act_fcn': 'Tanh', 'layers': 2, 'nodes': 12,
             'learning rate': 0.01},
            {'class_acc': 35.08771929824561, 'error': 1.9276482607088603, 'act_fcn': 'Tanh', 'layers': 4, 'nodes': 12,
             'learning rate': 0.01},
            {'class_acc': 35.526315789473685, 'error': 1.864608146460624, 'act_fcn': 'Tanh', 'layers': 10, 'nodes': 12,
             'learning rate': 0.01},
            {'class_acc': 36.40350877192983, 'error': 2.2569579230566195, 'act_fcn': 'ReLU', 'layers': 1, 'nodes': 12,
             'learning rate': 0.01},
            {'class_acc': 35.08771929824561, 'error': 1.7829300725233375, 'act_fcn': 'Sigmoid', 'layers': 2, 'nodes': 24,
             'learning rate': 0.01},
            {'class_acc': 36.40350877192983, 'error': 1.729126169781602, 'act_fcn': 'Tanh', 'layers': 4, 'nodes': 24,
             'learning rate': 0.01},
            {'class_acc': 38.59649122807017, 'error': 2.451741386039224, 'act_fcn': 'ReLU', 'layers': 1, 'nodes': 24,
             'learning rate': 0.01},
            {'class_acc': 35.96491228070175, 'error': 1.8330078038589717, 'act_fcn': 'ReLU', 'layers': 3, 'nodes': 24,
             'learning rate': 0.01},
            {'class_acc': 36.40350877192983, 'error': 1.8183778793903882, 'act_fcn': 'Tanh', 'layers': 2, 'nodes': 36,
             'learning rate': 0.01},
            {'class_acc': 37.280701754385966, 'error': 1.8193021998936225, 'act_fcn': 'Tanh', 'layers': 4, 'nodes': 36,
             'learning rate': 0.01},
            {'class_acc': 35.08771929824561, 'error': 1.8287659703915025, 'act_fcn': 'Tanh', 'layers': 5, 'nodes': 36,
             'learning rate': 0.01},
            {'class_acc': 37.719298245614034, 'error': 1.9769266347937682, 'act_fcn': 'ReLU', 'layers': 2, 'nodes': 36,
             'learning rate': 0.01},
            {'class_acc': 35.96491228070175, 'error': 1.9411223517122254, 'act_fcn': 'ReLU', 'layers': 4, 'nodes': 36,
             'learning rate': 0.01},
            {'class_acc': 40.35087719298245, 'error': 2.2013836396964055, 'act_fcn': 'ReLU', 'layers': 5, 'nodes': 36,
             'learning rate': 0.01},
            {'class_acc': 36.84210526315789, 'error': 1.757626810834704, 'act_fcn': 'Tanh', 'layers': 1, 'nodes': 48,
             'learning rate': 0.01},
            {'class_acc': 35.526315789473685, 'error': 1.7654175006501256, 'act_fcn': 'Tanh', 'layers': 3, 'nodes': 48,
             'learning rate': 0.01},
            {'class_acc': 36.40350877192983, 'error': 1.7956623320844742, 'act_fcn': 'Tanh', 'layers': 4, 'nodes': 48,
             'learning rate': 0.01},
            {'class_acc': 38.59649122807017, 'error': 2.0387485373553367, 'act_fcn': 'Tanh', 'layers': 5, 'nodes': 48,
             'learning rate': 0.01},
            {'class_acc': 35.526315789473685, 'error': 1.7836821381889076, 'act_fcn': 'Tanh', 'layers': 10, 'nodes': 48,
             'learning rate': 0.01},
            {'class_acc': 35.526315789473685, 'error': 1.8930196023202346, 'act_fcn': 'ReLU', 'layers': 2, 'nodes': 48,
             'learning rate': 0.01},
            {'class_acc': 35.96491228070175, 'error': 2.7204502554129055, 'act_fcn': 'ReLU', 'layers': 3, 'nodes': 48,
             'learning rate': 0.01},
            {'class_acc': 35.96491228070175, 'error': 2.1662045753265047, 'act_fcn': 'ReLU', 'layers': 5, 'nodes': 48,
             'learning rate': 0.01},
            {'class_acc': 35.96491228070175, 'error': 1.7941133089826493, 'act_fcn': 'Sigmoid', 'layers': 1, 'nodes': 12,
             'learning rate': 0.005},
            {'class_acc': 36.84210526315789, 'error': 1.7515484394595235, 'act_fcn': 'Sigmoid', 'layers': 2, 'nodes': 12,
             'learning rate': 0.005},
            {'class_acc': 40.35087719298245, 'error': 1.7176520429458662, 'act_fcn': 'Tanh', 'layers': 5, 'nodes': 12,
             'learning rate': 0.005},
            {'class_acc': 37.280701754385966, 'error': 1.827119109970516, 'act_fcn': 'Tanh', 'layers': 25, 'nodes': 12,
             'learning rate': 0.005},
            {'class_acc': 37.280701754385966, 'error': 1.7243996382335656, 'act_fcn': 'Tanh', 'layers': 1, 'nodes': 24,
             'learning rate': 0.005},
            {'class_acc': 38.15789473684211, 'error': 1.6918815173688981, 'act_fcn': 'Tanh', 'layers': 3, 'nodes': 24,
             'learning rate': 0.005},
            {'class_acc': 35.96491228070175, 'error': 1.69310226067008, 'act_fcn': 'Tanh', 'layers': 4, 'nodes': 24,
             'learning rate': 0.005},
            {'class_acc': 37.280701754385966, 'error': 1.8252131001233156, 'act_fcn': 'Tanh', 'layers': 5, 'nodes': 24,
             'learning rate': 0.005},
            {'class_acc': 35.526315789473685, 'error': 1.9419770200843387, 'act_fcn': 'ReLU', 'layers': 2, 'nodes': 24,
             'learning rate': 0.005},
            {'class_acc': 37.719298245614034, 'error': 1.759726846135118, 'act_fcn': 'ReLU', 'layers': 3, 'nodes': 24,
             'learning rate': 0.005},
            {'class_acc': 35.08771929824561, 'error': 3.2029149008421967, 'act_fcn': 'ReLU', 'layers': 4, 'nodes': 24,
             'learning rate': 0.005},
            {'class_acc': 39.91228070175439, 'error': 1.7449093645374352, 'act_fcn': 'Tanh', 'layers': 5, 'nodes': 36,
             'learning rate': 0.005},
            {'class_acc': 35.08771929824561, 'error': 1.786854655642375, 'act_fcn': 'Tanh', 'layers': 25, 'nodes': 36,
             'learning rate': 0.005},
            {'class_acc': 37.280701754385966, 'error': 2.034802322401958, 'act_fcn': 'ReLU', 'layers': 1, 'nodes': 36,
             'learning rate': 0.005},
            {'class_acc': 37.280701754385966, 'error': 2.123723648118889, 'act_fcn': 'ReLU', 'layers': 2, 'nodes': 36,
             'learning rate': 0.005},
            {'class_acc': 39.03508771929825, 'error': 2.0576942279220756, 'act_fcn': 'ReLU', 'layers': 3, 'nodes': 36,
             'learning rate': 0.005},
            {'class_acc': 36.40350877192983, 'error': 1.7591675514558598, 'act_fcn': 'Tanh', 'layers': 1, 'nodes': 48,
             'learning rate': 0.005},
            {'class_acc': 37.280701754385966, 'error': 1.709998317345013, 'act_fcn': 'Tanh', 'layers': 2, 'nodes': 48,
             'learning rate': 0.005},
            {'class_acc': 37.280701754385966, 'error': 1.7403800105077256, 'act_fcn': 'Tanh', 'layers': 10, 'nodes': 48,
             'learning rate': 0.005},
            {'class_acc': 35.96491228070175, 'error': 1.8301003773799873, 'act_fcn': 'Tanh', 'layers': 25, 'nodes': 48,
             'learning rate': 0.005},
            {'class_acc': 40.35087719298245, 'error': 1.7865848210436184, 'act_fcn': 'ReLU', 'layers': 4, 'nodes': 48,
             'learning rate': 0.005},
            {'class_acc': 36.40350877192983, 'error': 1.797073482081983, 'act_fcn': 'Sigmoid', 'layers': 1, 'nodes': 12,
             'learning rate': 0.001},
            {'class_acc': 35.526315789473685, 'error': 1.7288320945008067, 'act_fcn': 'Tanh', 'layers': 2, 'nodes': 12,
             'learning rate': 0.001},
            {'class_acc': 35.08771929824561, 'error': 2.0256506375196106, 'act_fcn': 'ReLU', 'layers': 1, 'nodes': 12,
             'learning rate': 0.001},
            {'class_acc': 36.84210526315789, 'error': 1.9764260467327246, 'act_fcn': 'ReLU', 'layers': 5, 'nodes': 24,
             'learning rate': 0.001},
            {'class_acc': 35.08771929824561, 'error': 1.7973611695124512, 'act_fcn': 'Sigmoid', 'layers': 1, 'nodes': 36,
             'learning rate': 0.001},
            {'class_acc': 36.40350877192983, 'error': 1.6793401809937971, 'act_fcn': 'Tanh', 'layers': 3, 'nodes': 36,
             'learning rate': 0.001},
            {'class_acc': 37.280701754385966, 'error': 1.6866835596497594, 'act_fcn': 'Tanh', 'layers': 4, 'nodes': 36,
             'learning rate': 0.001},
            {'class_acc': 39.03508771929825, 'error': 1.710536593908229, 'act_fcn': 'Tanh', 'layers': 10, 'nodes': 36,
             'learning rate': 0.001},
            {'class_acc': 35.08771929824561, 'error': 1.7333949695032613, 'act_fcn': 'Tanh', 'layers': 20, 'nodes': 36,
             'learning rate': 0.001},
            {'class_acc': 35.08771929824561, 'error': 1.887576546722105, 'act_fcn': 'ReLU', 'layers': 1, 'nodes': 36,
             'learning rate': 0.001},
            {'class_acc': 35.08771929824561, 'error': 1.8804585886624279, 'act_fcn': 'ReLU', 'layers': 3, 'nodes': 36,
             'learning rate': 0.001},
            {'class_acc': 36.40350877192983, 'error': 1.8599054127706458, 'act_fcn': 'ReLU', 'layers': 10, 'nodes': 36,
             'learning rate': 0.001},
            {'class_acc': 35.526315789473685, 'error': 1.7677314160094988, 'act_fcn': 'Tanh', 'layers': 1, 'nodes': 48,
             'learning rate': 0.001},
            {'class_acc': 35.526315789473685, 'error': 1.7410028132748703, 'act_fcn': 'Tanh', 'layers': 2, 'nodes': 48,
             'learning rate': 0.001},
            {'class_acc': 35.08771929824561, 'error': 1.7585955427367346, 'act_fcn': 'Tanh', 'layers': 3, 'nodes': 48,
             'learning rate': 0.001},
            {'class_acc': 35.526315789473685, 'error': 1.7373050191648973, 'act_fcn': 'Tanh', 'layers': 4, 'nodes': 48,
             'learning rate': 0.001},
            {'class_acc': 39.473684210526315, 'error': 1.6986077367166732, 'act_fcn': 'Tanh', 'layers': 10, 'nodes': 48,
             'learning rate': 0.001},
            {'class_acc': 35.96491228070175, 'error': 1.791240077342432, 'act_fcn': 'Tanh', 'layers': 20, 'nodes': 48,
             'learning rate': 0.001},
            {'class_acc': 37.280701754385966, 'error': 1.7508329593053267, 'act_fcn': 'Tanh', 'layers': 25, 'nodes': 48,
             'learning rate': 0.001},
            {'class_acc': 35.96491228070175, 'error': 1.8192759162204253, 'act_fcn': 'ReLU', 'layers': 2, 'nodes': 48,
             'learning rate': 0.001},
            {'class_acc': 38.59649122807017, 'error': 1.9284468398929164, 'act_fcn': 'ReLU', 'layers': 3, 'nodes': 48,
             'learning rate': 0.001},
            {'class_acc': 36.40350877192983, 'error': 1.9871552239545265, 'act_fcn': 'ReLU', 'layers': 10, 'nodes': 48,
             'learning rate': 0.001}]

        new_results = []
        best_params = {}
        best_params["class_acc"] = 0
        top_20 = sorted(good_models, key = lambda x: x["class_acc"], reverse=True)[:20]
        experiments = 0
        for trial in top_20:
            experiments += 1
            random.seed(100)
            np.random.seed(100)
            layers = trial["layers"]
            nodes = trial["nodes"]
            model = NeuralNetwork(input_nodes=X_train.shape[1],
                                  output_nodes=classes,
                                  output_activation=utils.softmax,
                                  output_deriv=utils.softmax_deriv,
                                  objective_fcn=utils.multiclass_crossentropy,
                                  learning_rate=trial["learning rate"])
            i = 0
            if trial["act_fcn"] == "Sigmoid":
                while i < layers:
                    model.add_layer(Layer(nodes=nodes,
                                          activation=utils.sigmoid,
                                          activation_deriv=utils.sigmoid_deriv))
                    i += 1
            elif trial["act_fcn"] == "Tanh":
                while i < layers:
                    model.add_layer(Layer(nodes=nodes,
                                          activation=utils.tanh,
                                          activation_deriv=utils.tanh_deriv))
                    i += 1
            elif trial["act_fcn"] == "ReLU":
                while i < layers:
                    model.add_layer(Layer(nodes=nodes,
                                          activation=utils.relu,
                                          activation_deriv=utils.relu_deriv))
                    i += 1

            model.fit(X_train,
                      y_train.T,
                      X_val,
                      y_val.T,
                      epochs=500,
                      include_bias=True,
                      early_stopping=True,
                      patience=3,
                      plot_name="Emotions Multiclass Classification\nGender Split\n" + str(layers) + " Hidden Layers, with " + str(
                          nodes) + " Nodes,\n" + trial["act_fcn"] + " Activation and " + str(
                          trial["learning rate"]) + " Learning Rate")

            class_acc, error = model.evaluate(X_test,
                                              y_test.T)

            current_model = {}
            current_model["class_acc"] = class_acc
            current_model["error"] = error
            current_model["act_fcn"] = trial["act_fcn"]
            current_model["layers"] = layers
            current_model["nodes"] = nodes
            current_model["model"] = model
            current_model["learning rate"] = trial["learning rate"]
            new_results.append(current_model.copy())
            print("model added")

            if class_acc > best_params["class_acc"]:
                best_params["class_acc"] = class_acc
                best_params["error"] = error
                best_params["act_fcn"] = trial["act_fcn"]
                best_params["layers"] = layers
                best_params["nodes"] = nodes
                best_params["model"] = model
                best_params["learning rate"] = trial["learning rate"]

            print(str(layers) + " Hidden Layers, with " + trial["act_fcn"] + " Activation")
            print("Test mean error:", error)
            print("Test classification accuracy:", class_acc, "\n\n")

        try:
            top_5 = pkl.load(open(Path("top_5_models.pkl"), "rb"))
        except:
            top_5 = sorted(new_results, key=lambda x: x["class_acc"], reverse=True)[:5]
            pkl.dump(top_5, open(Path("top_5_models.pkl"), "wb"))

        if not "final_analysis":
            top_patience_2 = pkl.load(open(Path("top_5_models_2.pkl"), "rb"))[0]
            top_patience_3 = pkl.load(open(Path("top_5_models.pkl"), "rb"))[0]
            top_patience_4 = pkl.load(open(Path("top_5_models_4.pkl"), "rb"))[0]

            # Moved the best performing model below so its data can be
            # used in subsequent confusion matrix.
            for model in [top_patience_2, top_patience_4, top_patience_3]:
                y_test_hat = model["model"].predict(X_test)
                y_folded = [x-8 if x >= 8 else x for x in np.argmax(y_test, axis=1)]
                y_hat_folded = [x - 8 if x >= 8 else x for x in y_test_hat]
                correct = [1 if y_i == y_hat_i else 0 for y_i, y_hat_i in zip(y_folded, y_hat_folded)]
                print("Folded accuracy is ", sum(correct)/len(correct))


            y_hat = top_patience_3["model"].predict(X_test)
            y_test_class = np.argmax(y_test, axis=1)

            conf = confusion_matrix(y_test_class, y_hat)
            if classes == 16:
                labels_male = [data.emotion_map[x + 1] + "_male" for x in range(int(classes / 2))]
                labels_female = [data.emotion_map[x + 1] + "_female" for x in range(int(classes / 2))]
                labels = labels_male + labels_female
            else:
                labels = [data.emotion_map[x + 1] for x in range(int(classes))]
            sn.heatmap(conf, cmap="Blues", annot=True,
                        xticklabels=labels, yticklabels=labels)
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title("Confusion Matrix for Split Classes")
            plt.show()
            plt.close()

            conf = confusion_matrix(y_folded, y_hat_folded)
            labels = [data.emotion_map[x + 1] for x in range(8)]
            sn.heatmap(conf, cmap="Blues", annot=True,
                        xticklabels=labels, yticklabels=labels)
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title("Confusion Matrix for Folded Classes")
            plt.show()


            # print("Best model:")
            # print("\tAccuracy:", best_params["class_acc"])
            # print("\tMean error:", best_params["error"])
            # print("\tActivation:", best_params["act_fcn"])
            # print("\tNumber of Layers:", best_params["layers"])
            # print("\tNumber of Nodes:", best_params["nodes"])
            # print("\tLearning Rate:", best_params["learning rate"])


if __name__ == "__main__":
    #regression_experiment()
    #binary_classification_experiment()
    #multiclass_classification_experiment()
    emotions_experiment()