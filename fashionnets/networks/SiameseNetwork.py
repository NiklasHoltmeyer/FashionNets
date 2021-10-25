from tensorflow.keras import Model
from tensorflow.keras import layers

from fashionnets.metrics.loss_layers import TripletLoss, QuadrupletLoss


class SiameseNetwork:
    @staticmethod
    def build(back_bone, triplets, input_shape, alpha, beta,
              preprocess_input=None, verbose=False, channels=3, **kwargs):
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

        if preprocess_input:
            encodings = [back_bone(preprocess_input(i)) for i in input_layers]
        else:
            encodings = [back_bone(i) for i in input_layers]

        output_layers = loss_layer(*encodings)

        return Model(
            inputs=input_layers, outputs=output_layers
        )
