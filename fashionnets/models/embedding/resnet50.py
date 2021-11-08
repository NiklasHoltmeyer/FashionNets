import tensorflow as tf
from fashionnets.models.layer.ResNetIdentityBlock import ResnetIdentityBlock
from tensorflow.keras.applications import resnet
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras.models import Sequential


class ResNet50Builder:
    @staticmethod
    def build(input_shape, embedding_dim=2048, weights="imagenet"):
        R = resnet.ResNet50(
            weights=weights, input_shape=input_shape + (3,), include_top=False
        )

        embedding_model = Sequential([R,
                                      #                                      tf.keras.layers.MaxPooling2D(),
                                      #                                      tf.keras.layers.Dropout(0.1),
                                      #                                      tf.keras.layers.BatchNormalization(name="bn2c"),
                                      tf.keras.layers.Conv2D(256, (3, 3), activation='relu',
                                                             kernel_initializer='he_uniform', padding='same',
                                                             kernel_regularizer=l2(2e-4)),
                                      ResnetIdentityBlock(3, [64, 64, 256]),
                                      tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'),
                                      tf.keras.layers.Flatten(),
                                      tf.keras.layers.Dense(embedding_dim, activation=None, use_bias=False)
                                      ])

        return embedding_model, resnet.preprocess_input

    @staticmethod
    def freeze_non_conv5_block1_out(model):
        print(f"Freeze non Conv5_Block1_Out Layers!")

        trainable = False
        for layer in model.layers:
            if layer.name == "conv5_block1_out":
                trainable = True
            layer.trainable = trainable

        for layer in model.layers[-20:]:
            layer.trainable = True

        return model

    @staticmethod
    def freeze_first_n_layers(model, n):
        print(f"Freeze first {n} Layers!")
        for l_idx, layer in enumerate(model.layers):
            layer.trainable = l_idx > n
        return model
