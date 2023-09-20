import tensorflow as tf


def compute_total_loss(logits, labels):
    """
    Computes the total loss
    
    Arguments:
    logits -- output of forward propagation (output of the last LINEAR unit),
        of shape (6, num_examples)
    labels -- "true" labels vector, same shape as Z3
    
    Returns:
    total_loss - Tensor of the total loss value
    """
    
    total_loss = tf.reduce_sum(
        tf.keras.losses.categorical_crossentropy(tf.transpose(labels),
                                                 tf.transpose(logits),
                                                 from_logits=True)
    )

    return total_loss
