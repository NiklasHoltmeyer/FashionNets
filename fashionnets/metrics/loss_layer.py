import tensorflow as tf
from fashionnets.metrics.distance_layers import TripletDistance, QuadrupletDistance
from tensorflow.keras import layers


# Triplet loss = AP - AN + alpha

class TripletLoss(layers.Layer):
    def __init__(self, alpha, **kwargs):
        super(TripletLoss, self).__init__(**kwargs)
        self.alpha = alpha
        self.alpha = 1.0
        self.distance_layer = TripletDistance()

    def call(self, anchor, positive, negative):
        ap_distance, an_distance = self.distance_layer(anchor, positive, negative)

        loss = ap_distance - an_distance + self.alpha
        loss = tf.maximum(loss, 0.0)

        return loss


# Quadruplet Loss = AP-AN + alpha_1 + AP-NN + alpha_2

class QuadrupletLoss(layers.Layer):
    def __init__(self, alpha, beta, skip_distance_layer=False, **kwargs):
        super(QuadrupletLoss, self).__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta

        self.distance_layer = QuadrupletDistance()

        if skip_distance_layer:
            self.distance_layer = lambda a, p, n, n2: (a, p, n, n2)

    def call(self, anchor, positive, negative1, negative2):
        # noinspection PyTupleAssignmentBalance
        ap_distance, an_distance, nn_distance = self.distance_layer(anchor, positive, negative1, negative2)

        loss = ap_distance - an_distance + self.alpha + ap_distance - nn_distance + self.beta

        loss = tf.maximum(loss, 0.0)

        return loss