import tensorflow as tf
from tensorflow.keras import layers


# Triplet loss = AP - AN + alpha

@tf.function
def euclidean_distance_sqrt(x, y):  # -> pow2 -> square
    return tf.sqrt(euclidean_distance(x, y))


@tf.function
def mean_euclidean_distance(x, y):
    return tf.reduce_mean(euclidean_distance(x, y))


@tf.function
def minkowski_distance(x, y):
    return tf.sqrt(tf.reduce_sum(l2_distance(x, y)))


@tf.function
def euclidean_distance2(x, y):
    sum_square = tf.math.reduce_sum(l2_distance(x, y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))


@tf.function
def l2_distance(x, y):
    return tf.math.square(tf.subtract(x, y))


@tf.function
def euclidean_distance(x, y):
    return tf.math.reduce_sum(l2_distance(x, y), axis=-1)


class TripletDistance(layers.Layer):
    def __init__(self, **kwargs):
        super(TripletDistance, self).__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap = euclidean_distance(anchor, positive)
        an = euclidean_distance(anchor, negative)

        return ap, an

class TripletCTLDistance(layers.Layer):
    def __init__(self, **kwargs):
        super(TripletCTLDistance, self).__init__(**kwargs)

    def call(self, anchor, positive_centroid, negative_centroid):
        print(anchor)
        print("*")
        print(positive_centroid)
        exit(0)
        a_cp = euclidean_distance(anchor, positive_centroid)
        a_cn = euclidean_distance(anchor, negative_centroid)

        return a_cp, a_cn

class QuadrupletDistance(layers.Layer):
    def __init__(self, **kwargs):
        super(QuadrupletDistance, self).__init__(**kwargs)

    def call(self, anchor, positive, negative1, negative2):
        ap = euclidean_distance(anchor, positive)
        an = euclidean_distance(anchor, negative1)
        nn = euclidean_distance(negative2, negative1)

        return ap, an, nn

class QuadrupletCTLDistance(layers.Layer):
    def __init__(self, **kwargs):
        super(QuadrupletCTLDistance, self).__init__(**kwargs)

    def call(self, anchor, negative1, positive_centroid, negative1_centroid, negative2_centroid):
        a_cp = euclidean_distance(anchor, positive_centroid)
        a_cn = euclidean_distance(anchor, negative1_centroid)
        n1_cn2 = euclidean_distance(negative1, negative2_centroid)

        return a_cp, a_cn, n1_cn2



