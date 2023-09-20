from linear_activation_forward import linear_activation_forward


def l_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID
    computation
    
    Arguments:
        X: data, numpy array of shape (input size, number of examples)
        parameters: output of initialize_parameters_deep()
    
    Returns:
        AL -- activation value from the output (last) layer
        caches -- list of caches containing every cache of
            linear_activation_forward() (there are L of them, indexed from 0 to
            L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    # The for loop starts at 1 because layer 0 is the input
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev,
                                             parameters['W' + str(l)],
                                             parameters['b' + str(l)],
                                             "relu")
        caches.append(cache)
    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A,
                                          parameters['W' + str(L)],
                                          parameters['b' + str(L)],
                                          "sigmoid")
    caches.append(cache)
          
    return AL, caches
