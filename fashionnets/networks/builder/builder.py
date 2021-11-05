from fashionnets.metrics.distance_layers import TripletDistance, QuadrupletDistance
from fashionnets.metrics.loss_layer import QuadrupletLoss, TripletLoss
from tensorflow.keras import Model
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers


def build_layers(builder):
    if builder.verbose:
        print(f"is_triplet={builder.is_triplet}, alpha={builder.alpha}, beta={builder.beta}")
    anchor_input = layers.Input(name="anchor", shape=builder._input_shape + (builder.channels,))
    positive_input = layers.Input(name="positive", shape=builder._input_shape + (builder.channels,))

    if builder.is_triplet:
        negative_inputs = [layers.Input(name="negative", shape=builder._input_shape + (builder.channels,))]
        distance_layer = TripletDistance()
        loss_layer = TripletLoss(alpha=builder.alpha)
    else:
        negative_inputs = [
            layers.Input(name="negative1", shape=builder._input_shape + (builder.channels,)),
            layers.Input(name="negative2", shape=builder._input_shape + (builder.channels,))
        ]
        distance_layer = QuadrupletDistance()
        loss_layer = QuadrupletLoss(alpha=builder.alpha, beta=builder.beta)

    input_layers = [anchor_input, positive_input, *negative_inputs]

    if builder.preprocess_input:
        encoding_layers = [builder.back_bone(builder.preprocess_input(i)) for i in input_layers]
    else:
        assert False, "Currently just using ResNet which requires Preprocessing"
        encoding_layers = [builder.back_bone(i) for i in input_layers]

    distance_layers = distance_layer(*encoding_layers)

    return input_layers, encoding_layers, distance_layers, loss_layer
