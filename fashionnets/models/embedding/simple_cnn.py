from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.python.ops import nn


class SimpleCNN:
    @staticmethod
    def build(input_shape, embedding_dim=256):
        model = keras.models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(input_shape[0], input_shape[1], 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu',
                               kernel_regularizer=l2(1e-3),
                               kernel_initializer='he_uniform'))
        model.add(layers.Dense(embedding_dim, activation=None,
                               kernel_regularizer=l2(1e-3),
                               kernel_initializer='he_uniform'))
        model.add(layers.Lambda(lambda x: nn.l2_normalize(x, axis=-1)))

        return model
