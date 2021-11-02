import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet


class ResNet50Builder:
    @staticmethod
    def build(input_shape, embedding_dim=256, weights="imagenet"):
        back_bone = resnet.ResNet50(
            weights=weights, input_shape=input_shape + (3,), include_top=False
        )

        x = tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu')(back_bone.output)
        x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu')(x)
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

##    @staticmethod
##    def build2(target_shape):
##        weights=None #"imagenet"
##        base_cnn = resnet.ResNet50(
##            weights=weights, input_shape=target_shape + (3,), include_top=False
##        )

##        flatten = tf.keras.layers.Flatten()(base_cnn.output)
##        dense1 = tf.keras.layers.Dense(512, activation="relu")(flatten)
##        dense1 = tf.keras.layers.BatchNormalization()(dense1)
##        dense2 = tf.keras.layers.Dense(256, activation="relu")(dense1)
##        dense2 = tf.keras.layers.BatchNormalization()(dense2)
##        output = tf.keras.layers.Dense(256)(dense2)

##        embedding = Model(base_cnn.input, output, name="Embedding")

##        trainable = False
##        for layer in base_cnn.layers:
##            if layer.name == "conv5_block1_out":
##                trainable = True
##            layer.trainable = trainable

##        return embedding, resnet.preprocess_input
