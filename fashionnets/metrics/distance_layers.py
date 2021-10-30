import tensorflow as tf
from tensorflow.keras import layers


# Triplet loss = AP - AN + alpha

class TripletDistance(layers.Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletDistance, self).__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)

        return ap_distance, an_distance

        #loss = ap_distance - an_distance + self.alpha
        #loss = tf.maximum(loss, 0.0)

        #return tf.reduce_sum(loss, axis=0)


# TripletSemiHardLoss
# Quadrup Loss = AP-AN + alpha_1 + AP-NN + alpha_2

class QuadrupletDistance(layers.Layer):
    def __init__(self, alpha, beta, **kwargs):
        self.alpha = alpha
        self.beta = beta
        super(QuadrupletDistance, self).__init__(**kwargs)

    def call(self, anchor, positive, negative1, negative2):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative1), -1)
        nn_distance = tf.reduce_sum(tf.square(negative1 - negative2), -1)

        return ap_distance, an_distance, nn_distance
        #loss = tf.maximum(ap_distance - an_distance + self.alpha, 0.0) + \
               #tf.maximum(ap_distance - nn_distance - self.beta, 0.0)

        #return tf.reduce_sum(loss, axis=0)




