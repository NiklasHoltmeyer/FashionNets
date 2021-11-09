import tensorflow as tf

from fashionnets.networks.builder.DistanceSiameseNetwork import DistanceSiameseNetwork


# noinspection PyAbstractClass
class SiameseNetwork(tf.keras.Model):
    def __init__(self, back_bone, is_triplet, input_shape, alpha, beta,
                 preprocess_input=None, verbose=False, channels=3):
        super(SiameseNetwork, self).__init__()
        self.back_bone = back_bone
        self.is_triplet = is_triplet
        self.input_shape_ = input_shape
        self.alpha = alpha
        self.beta = beta
        self.preprocess_input = preprocess_input
        self.verbose = verbose
        self.channels = channels
        # MetricLearningSiameseNetwork  DistanceSiameseNetwork
        network_type = DistanceSiameseNetwork(back_bone=back_bone, is_triplet=is_triplet, input_shape=input_shape,
                                              alpha=alpha, beta=beta,
                                              preprocess_input=preprocess_input, verbose=verbose,
                                              channels=channels)

        self.full_model, self.embedding_model, self.feature_extractor = network_type.combine()

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
