from tensorflow.keras import Model
from tensorflow.keras import layers

from fashionnets.metrics.distance_layer import DistanceLayerTriplets, DistanceLayerQuadruplets
from fashionnets.models.embedding.resnet50 import ResNet50Builder


class SiameseNetwork:
    # https://keras.io/examples/vision/siamese_network/
    @staticmethod
    def build(back_bone, triplets, input_shape, preprocess_input=None, channels=3):
        anchor_input = layers.Input(name="anchor", shape=input_shape + (channels,))
        positive_input = layers.Input(name="positive", shape=input_shape + (channels,))

        negative_inputs = None
        if triplets:
            negative_inputs = [layers.Input(name="negative", shape=input_shape + (channels,))]
        else:
            negative_inputs = [layers.Input(name="negative1", shape=input_shape + (channels,)),
                               layers.Input(name="negative2", shape=input_shape + (channels,))]

        input_layers = [anchor_input, positive_input, *negative_inputs]
        output_layers = [anchor_input, positive_input, *negative_inputs]

        if preprocess_input:
            output_layers = map(preprocess_input, output_layers)
        output_layers = map(back_bone, output_layers)
        output_layers = list(output_layers)

        output_layers = DistanceLayerTriplets()(*output_layers) if triplets \
            else DistanceLayerQuadruplets()(*output_layers)

        return Model(
            inputs=input_layers, outputs=output_layers
        )

if __name__ == "__main__":
    target_shape = (144, 144)

    embedding, preprocess_input = ResNet50Builder.build(target_shape)
    SiameseNetwork.build(embedding, triplets=False, input_shape=target_shape, preprocess_input=preprocess_input, channels=3)

