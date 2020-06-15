# Amie Roten
# CS559: Term Project
# Experiments for
# Neural Network
# Implementation

import random
from pathlib import Path
import pickle as pkl

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import seaborn as sn

import data
import utils
from model import NeuralNetwork, Layer

np.set_printoptions(suppress=True)

# Exploratory PCA function.
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