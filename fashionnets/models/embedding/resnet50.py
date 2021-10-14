from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import resnet
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.python.ops import nn

class ResNet50Builder:
    @staticmethod
    def build(input_shape, embedding_dim=256, weights="imagenet"):
        back_bone = resnet.ResNet50(
            weights=weights, input_shape=input_shape + (3,), include_top=False
        )

        flatten = layers.Flatten()(back_bone.output)
        dense1 = layers.Dense(512, activation="relu")(flatten)
        dense1 = layers.BatchNormalization()(dense1)
        dense2 = layers.Dense(256, activation="relu")(dense1)
        dense2 = layers.BatchNormalization()(dense2)

        output = layers.Dense(embedding_dim, activation='relu',
                               kernel_regularizer=l2(1e-3),
                               kernel_initializer='he_uniform')(dense2)

        l2_output = layers.Lambda(lambda x: nn.l2_normalize(x, axis=-1))(output)

        embedding = Model(back_bone.input, l2_output, name="Embedding")

        trainable = False
        for layer in back_bone.layers:
            if layer.name == "conv5_block1_out":
                trainable = True
            layer.trainable = trainable

        return embedding, resnet.preprocess_input

if __name__ == "__main__":
    ResNet50Builder.build((144,144))
