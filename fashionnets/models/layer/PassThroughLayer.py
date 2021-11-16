import tensorflow as tf

class PassThroughLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return inputs