import numpy as np


def sigmoid(x):
    """
    Args:
        x: numpy array of numbers

    Returns: element-wise operation 1 / (1 + exp(-n)) for n in x
    """
    return 1 / (1 + np.exp(-x)), x


def relu(x):
    """
    Args:
        x: numpy array of numbers

    Returns: element-wise operation max(0, n) for n in x
    """

    return np.maximum(0, x), x
