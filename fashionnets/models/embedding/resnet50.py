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
        x = tf.keras.layers.Dense(embedding_dim, activation=None)(x)
        x = tf.keras.layers.Lambda(lambda d: tf.math.l2_normalize(d, axis=1))(x)

        embedding_model = Model(back_bone.input, x, name="Embedding")

        if weights:
            trainable = False

            for layer in back_bone.layers:
                if layer.name == "conv5_block1_out":
                    trainable = True
                layer.trainable = trainable

            return embedding_model, resnet.preprocess_input
        else:
            return embedding_model, None




if __name__ == "__main__":
    ResNet50Builder.build((144, 144))
