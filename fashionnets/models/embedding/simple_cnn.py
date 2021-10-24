import tensorflow as tf


class SimpleCNN:
    @staticmethod
    def build(input_shape, embedding_dim=256):
        embedding_model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu',
                                   input_shape=(input_shape[0], input_shape[1], 3)),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(embedding_dim, activation=None),  # No activation on final dense layer
            tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))  # L2 normalize embeddings
        ])
        preprocessing = None #DS is already normalized!
        return embedding_model, preprocessing