import tensorflow as tf


def one_hot_matrix(label, depth=6):
    """
    Computes the one hot encoding for a single label

    Arguments:
        label --  (int) Categorical labels
        depth --  (int) Number of different classes that label can take

    Returns:
         one_hot -- tf.Tensor A single-column matrix with the one hot encoding.
    """
    # (approx. 1 line)
    one_hot = tf.reshape(tf.one_hot(label, depth, axis=0), shape=[-1, ])
    return one_hot
