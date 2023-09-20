import numpy as np


def initialize_parameters_he(layers_dims):
    """
    Arguments:
        layers_dims: python array (list) containing the size of each layer.
    
    Returns:
        parameters: python dictionary containing your parameters
            "W1", "b1", ..., "WL", "bL":
            W1 -- (layers_dims[1], layers_dims[0])
            b1 -- (layers_dims[1], 1)
            ...
            WL -- (layers_dims[L], layers_dims[L-1])
            bL -- (layers_dims[L], 1)
    """
    
    np.random.seed(3)
    parameters = dict()
    L = len(layers_dims) - 1  # integer representing the number of layers
     
    for l in range(1, L + 1):
        parameters['W' + str(l)] = np.random.randn(
            layers_dims[l], layers_dims[l-1]) * np.sqrt(2/layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        # YOUR CODE ENDS HERE

    return parameters
