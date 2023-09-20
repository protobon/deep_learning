import numpy as np
from activation import sigmoid


def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def sigmoid_backward(dA, Z):
    sigmoid_Z = sigmoid(Z)
    dZ = dA * sigmoid_Z * (1 - sigmoid_Z)
    return dZ
