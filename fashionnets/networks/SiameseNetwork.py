from tensorflow.keras import Model
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers

from fashionnets.metrics.loss_layers import TripletLoss, QuadrupletLoss


# noinspection PyAbstractClass,PyMethodOverriding
class SiameseNetwork(tf.keras.Model):
    def __init__(self, back_bone, is_triplet, input_shape, alpha, beta,
              preprocess_input=None, verbose=False, channels=3):
        super(SiameseNetwork, self).__init__()
        self.back_bone = back_bone
        self.is_triplet = is_triplet
        self._input_shape = input_shape
        self.alpha = alpha
        self.beta = beta
        self.preprocess_input = preprocess_input
        self.verbose = verbose
        self.channels = channels

        self.input_layer, self.encoding_layers, self.loss_layer = self.build_layers()
        self.full_model, self.embedding_model, self.feature_extractor = self.combine()

    def build_layers(self):
        if self.verbose:
            print(f"is_triplet={self.is_triplet}, alpha={self.alpha}, beta={self.beta}")
        anchor_input = layers.Input(name="anchor", shape=self._input_shape + (self.channels,))
        positive_input = layers.Input(name="positive", shape=self._input_shape + (self.channels,))

        if self.is_triplet:
            negative_inputs = [layers.Input(name="negative", shape=self._input_shape + (self.channels,))]
            loss_layer = TripletLoss(alpha=self.alpha, name='triplet_loss_layer')
        else:
            negative_inputs = [
                layers.Input(name="negative1", shape=self._input_shape + (self.channels,)),
                layers.Input(name="negative2", shape=self._input_shape + (self.channels,))
            ]
            loss_layer = QuadrupletLoss(alpha=self.alpha, beta=self.beta, name='triplet_loss_layer')

        input_layers = [anchor_input, positive_input, *negative_inputs]

        if self.preprocess_input:
            encodings = [self.back_bone(self.preprocess_input(i)) for i in input_layers]
        else:
            encodings = [self.back_bone(i) for i in input_layers]

        return input_layers, encodings, loss_layer

    def combine(self):
        output_layers = self.loss_layer(*self.encoding_layers)

        full_model = Model(
            inputs=self.input_layer, outputs=output_layers
        )
        embedding_model = Model(inputs=self.input_layer, outputs=self.encoding_layers)

        input_layer = layers.Input(name="input_image", shape=self._input_shape + (self.channels,))
        encoding = self.back_bone(self.preprocess_input(input_layer))
        feature_extractor = Model(inputs=[input_layer], outputs=encoding)

        return full_model, embedding_model, feature_extractor

    def call(self, inputs):
        """
        Calculate Quad/Trip Loss
        :param inputs:
        :return:
        """
        return self.full_model(inputs)

    def embed(self, inputs):
        """
        Feature Vectors for N Input Images (Triplet -> N=3)
        :param inputs:
        :return:
        """
        return self.embedding_model(inputs)
#        return self.input_layer(inputs)

    def extract_features(self, image):
        """
        :param image: As Matrix
        :return: Feature-Vector
        """
        return self.feature_extractor(image)

