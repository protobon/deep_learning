import tensorflow as tf


def initialize_parameters():
    """
    Initializes parameters to build a neural network with TensorFlow.
    The shapes are:
                        W1 : [25, 12288]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [6, 12]
                        b3 : [6, 1]
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
    
    initializer = tf.keras.initializers.GlorotNormal(seed=1)
    W1 = tf.Variable(initializer(shape=(25, 12288)), name="W1")
    b1 = tf.Variable(initializer(shape=(25, 1)), name='b1')
    W2 = tf.Variable(initializer(shape=(12, 25)), name="W2")
    b2 = tf.Variable(initializer(shape=(12, 1)), name='b2')
    W3 = tf.Variable(initializer(shape=(6, 12)), name="W3")
    b3 = tf.Variable(initializer(shape=(6, 1)), name='b3')

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters
