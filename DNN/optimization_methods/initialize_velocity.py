import numpy as np


def initialize_velocity(parameters):
    """
        Initializes the velocity as a python dictionary with:
            - keys: "dW1", "db1", ..., "dWL", "dbL"
            - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    Args:
        parameters: python dictionary containing your parameters.
    Returns:
        v: python dictionary containing the current velocity.
    """
    
    L = len(parameters) // 2  # number of layers in the neural networks
    v = dict()
    
    # Initialize velocity
    for l in range(1, L + 1):
        v["dW" + str(l)] = np.zeros((parameters["W" + str(l)].shape[0],
                                     parameters["W" + str(l)].shape[1]))
        v["db" + str(l)] = np.zeros((parameters["b" + str(l)].shape[0],
                                     parameters["b" + str(l)].shape[1]))

    return v
