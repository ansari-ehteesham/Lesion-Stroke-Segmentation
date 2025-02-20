import tensorflow as tf


# 2D Unet Model Loss Function
def focal_tversky_loss(y_true, y_pred, alpha=0.7, gamma=0.75):
    # print("*"*50)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))

    tversky = (tp + 1e-6) / (tp + alpha * fp + (1 - alpha) * fn + 1e-6)

    return tf.pow(1 - tversky, gamma)