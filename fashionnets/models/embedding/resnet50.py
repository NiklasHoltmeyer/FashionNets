from tensorflow.keras import layers
from tensorflow.keras import layers
from tensorflow.keras.applications import resnet


class ResNet50Builder:
    @staticmethod
    def build(input_shape, weights="imagenet"):
        back_bone = resnet.ResNet50(
            weights=weights, input_shape=input_shape + (3,), include_top=False
        )

        flatten = layers.Flatten()(back_bone.output)
        dense1 = layers.Dense(512, activation="relu")(flatten)
        dense1 = layers.BatchNormalization()(dense1)
        dense2 = layers.Dense(256, activation="relu")(dense1)
        dense2 = layers.BatchNormalization()(dense2)
        output = layers.Dense(256)(dense2)

        embedding = Model(back_bone.input, output, name="Embedding")

        trainable = False
        for layer in back_bone.layers:
            if layer.name == "conv5_block1_out":
                trainable = True
            layer.trainable = trainable

        return embedding, resnet.preprocess_input


