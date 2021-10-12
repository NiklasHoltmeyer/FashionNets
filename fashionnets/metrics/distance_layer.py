import tensorflow as tf
from tensorflow.keras import layers


class DistanceLayerTriplets(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return ap_distance, an_distance


class DistanceLayerQuadruplets(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative1, negative2):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an1_distance = tf.reduce_sum(tf.square(anchor - negative1), -1)
        an2_distance = tf.reduce_sum(tf.square(anchor - negative2), -1)
        return ap_distance, an1_distance, an2_distance
