import tensorflow as tf


class RandomHorizontalFlip(tf.keras.layers.Layer):
    def __init__(self, p):
        super(RandomHorizontalFlip, self).__init__()
        self.probability = p

    def call(self, images):
        if tf.random.uniform([]) < self.probability:
            return tf.image.flip_left_right(images)
        return images
