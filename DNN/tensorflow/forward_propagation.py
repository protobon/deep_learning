import tensorflow as tf


def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model:
        LINEAR -> RELU -> LINEAR -> RELU -> LINEAR
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2",
        "b2", "W3", "b3" the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    Z1 = tf.matmul(W1, X) + b1          # Z1 = np.dot(W1, X) + b1
    A1 = tf.keras.activations.relu(Z1)  # A1 = relu(Z1)
    Z2 = tf.matmul(W2, A1) + b2         # Z2 = np.dot(W2, A1) + b2
    A2 = tf.keras.activations.relu(Z2)  # A2 = relu(Z2)
    Z3 = tf.matmul(W3, A2) + b3         # Z3 = np.dot(W3, A2) + b3
    
    return Z3
