import numpy as np


def gradient_check(x, theta, epsilon=1e-7, print_msg=False) -> float:
    """
    Implement the gradient checking presented in Figure 1.
    
    Arguments:
        x: a float input
        theta: our parameter, a float as well
        epsilon: tiny shift to the input to compute approximated gradient
        print_msg: prints message if the gradients are wrong
    Returns:
        difference: difference between the approximated gradient and the
            backward propagation gradient.
    """
    
    theta_plus = theta + epsilon
    theta_minus = theta - epsilon
    J_plus = forward_propagation(x, theta_plus)
    J_minus = forward_propagation(x, theta_minus)
    gradapprox = (J_plus - J_minus) / (2 * epsilon)
    
    # Check if gradapprox is close enough to the output of backward_propagation()
    grad = backward_propagation(x, theta)
    
    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator / denominator

    if print_msg:
        if difference > 2e-7:
            print("\033[93m" +
                  "There is a mistake in the backward propagation!\n"
                  + f"diff: {str(difference)}" + "\033[0m")

    return difference
