import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet


class ResNet50Builder:
    @staticmethod
    def build(input_shape, embedding_dim=2048, weights="imagenet"):
        back_bone = resnet.ResNet50(
            weights=weights, input_shape=input_shape + (3,), include_top=False
        )

        x = tf.keras.layers.Conv2D(filters=128, kernel_size=2, padding='same', activation='relu')(back_bone.output)
        x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(embedding_dim)(x)
        #x = tf.keras.layers.Lambda(lambda d: tf.math.l2_normalize(d, axis=1))(x)

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


