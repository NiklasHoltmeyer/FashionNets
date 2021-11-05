import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet
from tensorflow.keras.regularizers import l2


class ResNet50Builder:
    @staticmethod
    def build(input_shape, embedding_dim=2048, weights="imagenet"):
        back_bone = resnet.ResNet50(
            weights=weights, input_shape=input_shape + (3,), include_top=False
        )

        x = tf.keras.layers.Conv2D(128, (7, 7), activation='relu', padding='same',
                                   input_shape=input_shape,
                                   kernel_initializer='he_uniform',
                                   kernel_regularizer=l2(2e-4))(back_bone.output)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                                   kernel_regularizer=l2(2e-4))(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(4096, activation='relu',
                                  kernel_regularizer=l2(1e-3),
                                  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.Dense(embedding_dim, activation=None,
                                  kernel_regularizer=l2(1e-3),
                                  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.Lambda(lambda d: tf.math.l2_normalize(d, axis=-1))(x) # L2 normalize embeddings

        embedding = Model(back_bone.input, x, name="Embedding")

        #        #if weights:
        #        trainable = False

        #        for layer in back_bone.layers:
        #            if layer.name == "conv5_block1_out":
        #                trainable = True
        #            layer.trainable = trainable

        return embedding, resnet.preprocess_input

    @staticmethod
    def freeze_non_conv5_block1_out(model):
        print(f"Freeze non Conv5_Block1_Out Layers!")

        trainable = False
        for layer in model.layers:
            if layer.name == "conv5_block1_out":
                trainable = True
            layer.trainable = trainable

        for layer in model.layers[-8:]:
            layer.trainable = True

        return model

    @staticmethod
    def freeze_first_n_layers(model, n):
        print(f"Freeze first {n} Layers!")
        for l_idx, layer in enumerate(model.layers):
            layer.trainable = l_idx > n
        return model
