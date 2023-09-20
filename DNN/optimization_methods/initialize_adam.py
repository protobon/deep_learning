import numpy as np


def initialize_adam(parameters):
    """
    Args:
        parameters: python dictionary containing your parameters.
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl

    Returns:
        v: python dictionary that will contain the exponentially weighted
        average of the gradient. Initialized with zeros.
        s: python dictionary that will contain the exponentially weighted
        average of the squared gradient. Initialized with zeros.
    """

    L = len(parameters) // 2  # number of layers in the neural networks
    v = dict()
    s = dict()
    
    # Initialize v, s. Input: "parameters". Outputs: "v, s".
    for l in range(1, L + 1):
        v["dW" + str(l)] = np.zeros((parameters["W" + str(l)].shape[0],
                                     parameters["W" + str(l)].shape[1]))
        v["db" + str(l)] = np.zeros((parameters["b" + str(l)].shape[0],
                                     parameters["b" + str(l)].shape[1]))
        s["dW" + str(l)] = np.zeros((parameters["W" + str(l)].shape[0],
                                     parameters["W" + str(l)].shape[1]))
        s["db" + str(l)] = np.zeros((parameters["b" + str(l)].shape[0],
                                     parameters["b" + str(l)].shape[1]))

    return v, s
