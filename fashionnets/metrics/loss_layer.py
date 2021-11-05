import tensorflow as tf
from fashionnets.metrics.distance_layers import TripletDistance, QuadrupletDistance
from tensorflow.keras import layers


# Triplet loss = AP - AN + alpha

class TripletLoss(layers.Layer):
    def __init__(self, alpha, **kwargs):
        super(TripletLoss, self).__init__(**kwargs)
        print("Triplet" * 100)
        self.alpha = alpha
        self.alpha = 1.0
        self.distance_layer = TripletDistance()

    def call(self, anchor, positive, negative):
        ap_distance, an_distance = self.distance_layer(anchor, positive, negative)

        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.alpha, 0.0)

        return loss


# Quadrup Loss = AP-AN + alpha_1 + AP-NN + alpha_2

class QuadrupletLoss(layers.Layer):
    def __init__(self, alpha, beta, skip_distance_layer=False, **kwargs):
        print("Quadloss" * 100)
        super(QuadrupletLoss, self).__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta

        self.distance_layer = QuadrupletDistance()

        if skip_distance_layer:
            self.distance_layer = lambda a, p, n, n2: (a, p, n, n2)

    def call(self, anchor, positive, negative1, negative2):
        ap, an, nn = self.distance_layer(anchor, positive, negative1, negative2)

        ap, an, nn = tf.square(ap), tf.square(an), tf.square(nn)

        ap_an = tf.math.reduce_sum(
            tf.maximum(
                (tf.subtract(ap, an) + self.alpha)
                , 0.0
            )
        )

        ap_nn = tf.math.reduce_sum(
            tf.maximum(
                (tf.subtract(ap, nn) + self.beta)
                , 0.0
            )
        )

        return ap_an + ap_nn
