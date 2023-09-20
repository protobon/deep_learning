import numpy as np
from forward_propagation_n import forward_propagation_n


def gradient_check_n(parameters, gradients, X, Y, epsilon=1e-7):
    """
    Checks if backward_propagation_n computes correctly the gradient of the
    cost output by forward_propagation_n
    
    Arguments:
        parameters: python dictionary containing your parameters
            "W1", "b1", "W2", "b2", "W3", "b3"
        gradients: output of backward_propagation_n, contains gradients of the
            cost with respect to the parameters
        X: input datapoint, of shape (input size, number of examples)
        Y: true "label"
        epsilon: tiny shift to the input to compute approximated gradient
    Returns:
    difference: difference between the approximated gradient and the backward
        propagation gradient
    """
    
    # Set-up variables
    parameters_values, _ = dictionary_to_vector(parameters)
    
    grad = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))
    
    # Compute gradapprox
    for i in range(num_parameters):
        
        # Compute J_plus[i].
        # Inputs: parameters_values, epsilon.
        # Output = J_plus[i].
        theta_plus = np.copy(parameters_values)
        theta_plus[i] += epsilon
        J_plus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(theta_plus))
        # YOUR CODE ENDS HERE
        
        # Compute J_minus[i].
        # Inputs: parameters_values, epsilon.
        # Output = J_minus[i].
        theta_minus = np.copy(parameters_values)
        theta_minus[i] -= epsilon
        J_minus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(theta_minus))

        # Compute gradapprox[i]
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)

    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator / denominator
    
    if difference > 2e-7:
        print("\033[93m" + "There is a mistake in the backward propagation!\n" +
              f"difference = {str(difference)}" + "\033[0m")

    return difference
