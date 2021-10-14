from tensorflow.keras import Model
from tensorflow.keras import layers

from fashionnets.metrics.loss_layers import TripletLoss, QuadrupletLoss
from fashionnets.models.embedding.resnet50 import ResNet50Builder


class SiameseNetwork:
    @staticmethod
    def build(back_bone, triplets, input_shape, alpha=1.0, beta=0.5, channels=3, verbose=False):
        if verbose:
            print(f"triplets={triplets}, alpha={alpha}, beta={beta}")
        anchor_input = layers.Input(name="anchor", shape=input_shape + (channels,))
        positive_input = layers.Input(name="positive", shape=input_shape + (channels,))

        if triplets:
            negative_inputs = [layers.Input(name="negative", shape=input_shape + (channels,))]
            loss_layer = TripletLoss(alpha=alpha, name='triplet_loss_layer')
        else:
            negative_inputs = [
                layers.Input(name="negative1", shape=input_shape + (channels,)),
                layers.Input(name="negative2", shape=input_shape + (channels,))
            ]
            loss_layer = QuadrupletLoss(alpha=alpha, beta=beta, name='triplet_loss_layer')

        input_layers = [anchor_input, positive_input, *negative_inputs]

        encodings = [back_bone(i) for i in input_layers]
        output_layers = loss_layer(*encodings)

        return Model(
            inputs=input_layers, outputs=output_layers
        )


if __name__ == "__main__":
    target_shape = (144, 144)

    embedding, preprocess_input = ResNet50Builder.build(target_shape)
    SiameseNetwork.build(embedding, triplets=True, input_shape=target_shape)
