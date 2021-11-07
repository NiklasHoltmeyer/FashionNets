import tensorflow as tf


class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters) -> object:
        super(ResnetIdentityBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1), name="conv2a")
        self.bn2a = tf.keras.layers.BatchNormalization(name="bn2a")

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same', name="conv2b")
        self.bn2b = tf.keras.layers.BatchNormalization(name="bn2b")

        self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1), name="conv2c")
        self.bn2c = tf.keras.layers.BatchNormalization(name="bn2c")

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        x += input_tensor
        return tf.nn.relu(x)
