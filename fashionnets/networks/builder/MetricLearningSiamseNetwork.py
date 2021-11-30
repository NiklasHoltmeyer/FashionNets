from fashiondatasets.utils.logger.defaultLogger import defaultLogger
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

from fashionnets.metrics.loss_layer import TripletLoss
from fashionnets.networks.builder.builder import build_layers

logger = defaultLogger("deepfashion_model_builder")

class MetricLearningSiameseNetwork:
    def __init__(self, back_bone, is_triplet, input_shape, alpha, beta,
                 preprocess_input=None, verbose=False, channels=3):
        self.back_bone = back_bone
        self.is_triplet = is_triplet
        self.input_shape = input_shape
        self.alpha = alpha
        self.beta = beta
        self.preprocess_input = preprocess_input
        self.verbose = verbose
        self.channels = channels

        self.input_layers, self.encoding_layers, self.distance_layers, self.loss_layer = build_layers(self)

    # noinspection PyUnboundLocalVariable
    def combine(self):
        embedding_model = Model(inputs=self.input_layers, outputs=self.encoding_layers)
        input_layer = layers.Input(name="input_image", shape=self.input_shape + (self.channels,))
        encoding = self.back_bone(self.preprocess_input(input_layer))
        feature_extractor = Model(inputs=[input_layer], outputs=encoding)

        encoded_a, encoded_p, encoded_n = self.encoding_layers[:3]

        if self.is_triplet:
            loss_layer = TripletLoss(self.alpha)(encoded_a, encoded_p, encoded_n)

            full_model = Model(inputs=self.input_layers, outputs=loss_layer)
            return full_model, embedding_model, feature_extractor

        if not self.is_triplet:  # <- Should always be True - but just in case
            encoded_n2 = self.encoding_layers[3]

        ap_encoded = layers.Concatenate(axis=-1, name="Anchor-Positive")([encoded_a, encoded_p])
        an_encoded = layers.Concatenate(axis=-1, name="Anchor-Negative")([encoded_a, encoded_n])

        if not self.is_triplet:
            # noinspection PyUnboundLocalVariable
            nn_encoded = layers.Concatenate(axis=-1, name="Negative-Negative2")([encoded_n, encoded_n2])

        metric_network = self.build_metric_network()

        ap_dist = metric_network(ap_encoded)
        an_dist = metric_network(an_encoded)
        nn_dist = metric_network(nn_encoded)

        output_layers = self.loss_layer([ap_dist, an_dist, nn_dist], skip_distance_layer=True)

        full_model = Model(
            inputs=self.input_layers, outputs=output_layers
        )

        return full_model, embedding_model, feature_extractor

    def build_metric_network(self):
        _input_shape = self.input_shape
        logger.debug(self.input_shape)
        _input_shape = self.input_shape[0] * 2, self.input_shape[1]

        network = keras.Sequential(name="learned_metric")
        network.add(layers.Dense(10, activation='relu',
                                 input_shape=_input_shape,
                                 kernel_regularizer=l2(1e-3),
                                 kernel_initializer='he_uniform'))
        network.add(layers.Dense(10, activation='relu',
                                 kernel_regularizer=l2(1e-3),
                                 kernel_initializer='he_uniform'))
        network.add(layers.Dense(10, activation='relu',
                                 kernel_regularizer=l2(1e-3),
                                 kernel_initializer='he_uniform'))

        # Last layer : binary softmax
        network.add(layers.Dense(2, activation='softmax'))

        # Select only one output value from the softmax
        network.add(layers.Lambda(lambda x: x[:, 0]))

        return network
