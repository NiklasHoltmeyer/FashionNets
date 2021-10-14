#import tensorflow as tf
#from tensorflow.keras import layers


#class DistanceLayerTriplets(layers.Layer):
#    def __init__(self, **kwargs):
#        super().__init__(**kwargs)

#    def call(self, anchor, positive, negative):
#        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
#        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
#        return ap_distance, an_distance

## Triplet loss = AP - AN + alpha
## Quadrup Loss = AP-AN + alpha_1 + AP-NN + alpha_2

## AP = dist(A, P)
## AN = dist(A, N_1)
## NN = dist(N_1, N_2)

#class DistanceLayerQuadruplets(layers.Layer):
#    def __init__(self, **kwargs):
#        super().__init__(**kwargs)

#    def call(self, anchor, positive, negative1, negative2):
#        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
#        an_distance = tf.reduce_sum(tf.square(anchor - negative1), -1)
#        nn_distance = tf.reduce_sum(tf.square(negative1 - negative2), -1)

#        return ap_distance, an_distance, nn_distance
