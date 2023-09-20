import numpy as np
from initialize_parameters_deep import initialize_parameters_deep
from L_model_forward import l_model_forward
from compute_cost import compute_cost
from L_model_backward import l_model_backward
from update_parameters import update_parameters


def l_layer_model(X,
                  Y,
                  layers_dims,
                  learning_rate=0.0075,
                  num_iterations=3000,
                  print_cost=False):
    """
    Implements an L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
        X: input data, of shape (n_x, number of examples)
        Y: true "label" vector (containing 1 if cat, 0 if non-cat), of shape
            (1, number of examples)
        layers_dims: list containing the input size and each layer size,
            of length (number of layers + 1).
        learning_rate: learning rate of the gradient descent update rule
        num_iterations: number of iterations of the optimization loop
        print_cost: if True, it prints the cost every 100 steps
    
    Returns:
        parameters: parameters learnt by the model. They can then be used to
            predict.
    """

    np.random.seed(1)
    costs = []  # keep track of cost
    
    parameters = initialize_parameters_deep(layers_dims)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):
        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = l_model_forward(X, parameters)
        
        # Compute cost.
        cost = compute_cost(AL, Y)
    
        # Backward propagation.
        grads = l_model_backward(AL, Y, caches)
 
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
                
        # Print the cost every 100 iterations
        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            print(f"Cost after iteration {i}: {np.squeeze(cost)}")
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)
    
    return parameters, costs
