import numpy as np


def update_parameters_with_adam(parameters, grads, v: dict, s: dict, t,
                                learning_rate=0.01, beta1=0.9, beta2=0.999,
                                epsilon=1e-8):
    """
    Update parameters using Adam
    
    Arguments:
        parameters: python dictionary containing your parameters
        grads: python dictionary containing your gradients for each parameter:
        v: Adam variable, moving average of the first gradient
        s: Adam variable, moving average of the squared gradient
        t: Adam variable, counts the number of taken steps
        learning_rate: the learning rate, scalar.
        beta1: Exponential decay hyperparameter for the first moment estimates
        beta2: Exponential decay hyperparameter for the second moment estimates
        epsilon: hyperparameter preventing division by zero in Adam updates

    Returns:
    parameters -- python dictionary containing your updated parameters 
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    """
    
    L = len(parameters) // 2  # number of layers in the neural networks
    v_corrected = dict()  # Initializing first moment estimate
    s_corrected = dict()  # Initializing second moment estimate
    
    # Perform Adam update on all parameters
    for l in range(1, L + 1):
        # Moving average of the gradients.
        # Inputs: "v, grads, beta1". Output: "v".
        v["dW" + str(l)] = beta1 * v["dW" + str(l)] + (
                (1 - beta1) * grads["dW" + str(l)])
        v["db" + str(l)] = beta1 * v["db" + str(l)] + (
                (1 - beta1) * grads["db" + str(l)])

        # Compute bias-corrected first moment estimate.
        # Inputs: "v, beta1, t". Output: "v_corrected".
        v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - beta1**t)
        v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - beta1**t)

        # Moving average of the squared gradients.
        # Inputs: "s, grads, beta2". Output: "s".
        s["dW" + str(l)] = (beta2 * s["dW" + str(l)]) + (
                (1 - beta2) * np.square(grads["dW" + str(l)]))
        s["db" + str(l)] = (beta2 * s["db" + str(l)]) + (
                (1 - beta2) * np.square(grads["db" + str(l)]))

        # Compute bias-corrected second raw moment estimate.
        # Inputs: s, beta2, t.
        # Output: "s_corrected".
        s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - beta2**t)
        s_corrected["db" + str(l)] = s["db" + str(l)] / (1 - beta2**t)

        # Update parameters.
        # Inputs: parameters, learning_rate, v_corrected, s_corrected, epsilon.
        # Output: parameters.
        parameters["W" + str(l)] -= learning_rate * v_corrected["dW" + str(l)] / \
                                    (np.sqrt(s_corrected["dW" + str(l)]) +
                                     epsilon)
        parameters["b" + str(l)] -= learning_rate * v_corrected["db" + str(l)] / \
                                    (np.sqrt(s_corrected["db" + str(l)]) +
                                     epsilon)

    return parameters, v, s, v_corrected, s_corrected
