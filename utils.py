import numpy as np
from keras import activations as act
from keras import losses as ls
from scipy.special import softmax as scipy_softmax
from sklearn.metrics import mean_squared_error, \
                            log_loss

####
###
##
# Utility functions for neural network.
# These are all expecting single observations
# as input.

# Similar to sigmoid:
#   -1 < x < 1
#
# Apparently, results in stronger
# derivatives, so tends to perform
# better.
def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

# 0 < x < 1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# x >= 0 --> x
# x < 0 --> 0
def relu(x):
    return np.maximum(x, np.zeros(x.shape[0]))

# All values in vector X sum to one, can
# be used to represent probabilities.
def softmax(x):
    return np.exp(x)/sum(np.exp(x))

# X --> X
def identity(x):
    return x

# Use for regression, single output
# with identity function as activation.
def MSE(y, y_hat):
    all_sq_errors = [(y_i - y_hat_i) ** 2 \
                     for y_i, y_hat_i in zip(y, y_hat)]
    return (1 / len(y)) * sum(all_sq_errors)

# Use for binary classification, single
# output with sigmoid activation.
def binary_crossentropy(y, y_hat):
    return -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

# Use for multi-class classification,
# outputs equal to the number of classes,
# with softmax as output activation.
def categorical_crossentropy(y, y_hat):
    return sum([binary_crossentropy(y_k, y_hat_k) \
                for y_k, y_hat_k in zip(y, y_hat)])


####
###
##
# Derivatives of each function.

def tanh_deriv(x):
    return 1 - tanh(x)^2

def sigmoid_deriv(x):
    return sigmoid(x)(1-sigmoid(x))

def relu_deriv(x):
    return np.where(x < 0, 0, 1)

def softmax_deriv(x):
    raise NotImplementedError

def identity_deriv(x):
    return np.ones(x.shape[1])

def MSE_deriv(y, y_hat):
    raise NotImplementedError

def binary_crossentropy_deriv(y, y_hat):
    raise NotImplementedError

if __name__ == "__main__":
    x = np.random.rand(10)
    y = np.random.rand(10)
    y_hat = np.random.rand(10)
    my_tanh = tanh(x)
    keras_tanh = act.tanh(x)
    assert np.allclose(my_tanh, np.array(keras_tanh))
    my_sigmoid = sigmoid(x)
    keras_sigmoid = act.sigmoid(x)
    assert np.allclose(my_sigmoid, np.array(keras_sigmoid))
    my_relu = relu(x)
    keras_relu = act.relu(x, alpha = 0.0)
    assert np.allclose(my_relu, np.array(keras_relu))
    my_softmax = softmax(x)
    scipy_softmax = scipy_softmax(x)
    assert np.allclose(my_softmax, scipy_softmax)
    my_MSE = MSE(y, y_hat)
    sklearn_MSE = mean_squared_error(y, y_hat)
    assert np.allclose(my_MSE, sklearn_MSE)

