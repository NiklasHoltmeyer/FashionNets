from fashionnets.models.embedding.simple_cnn import SimpleCNN
from fashionnets.networks.builder.builder import build_layers
from tensorflow.keras import Model
from tensorflow.keras import layers


class DistanceSiameseNetwork:
    def __init__(self, back_bone, is_triplet, is_ctl, input_shape, alpha, beta,
                 preprocess_input=None, verbose=False, channels=3):
        self.back_bone = back_bone
        self.is_triplet = is_triplet
        self.is_ctl = is_ctl
        self.input_shape = input_shape
        self.alpha = alpha
        self.beta = beta
        self.preprocess_input = preprocess_input
        self.verbose = verbose
        self.channels = channels

        self.input_layers, self.encoding_layers, self.distance_layers, self.loss_layer = build_layers(self)

    def combine(self):
        print((self.encoding_layers))
        print(len(self.encoding_layers))
        print(self.loss_layer)
        output_layers = self.loss_layer(*self.encoding_layers)

        full_model = Model(
            inputs=self.input_layers, outputs=output_layers
        )

        embedding_model = Model(inputs=self.input_layers, outputs=self.encoding_layers)

        input_layer = layers.Input(name="input_image", shape=self.input_shape + (self.channels,))

        encoding = self.back_bone(self.preprocess_input(input_layer))

        feature_extractor = Model(inputs=[input_layer], outputs=encoding)

        return full_model, embedding_model, feature_extractor

if __name__ == "__main__":
    class FakeBuilder:
        def __init__(self):
            self.back_bone, _ = SimpleCNN.build((224, 224), 2048)
            self.is_triplet = True
            self.is_ctl = True
            self.input_shape = (224, 224)
            self.alpha = 1.0
            self.beta = 0.5
            self.preprocess_input = lambda d: d
            self.verbose = True
            self.channels = 3


    dsn = DistanceSiameseNetwork(SimpleCNN.build((224, 224), 2048)[0], is_triplet=False, is_ctl=True,
                           input_shape=(224, 224), alpha=1.0, beta=0.5,
                 preprocess_input=lambda d: d, verbose=True, channels=3)

