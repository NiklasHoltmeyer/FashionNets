from tensorflow.keras import layers

from fashionnets.metrics.distance_layers import TripletDistance, QuadrupletDistance, TripletCTLDistance, \
    QuadrupletCTLDistance
from fashionnets.metrics.loss_layer import QuadrupletLoss, TripletLoss
from fashionnets.models.layer.PassThroughLayer import PassThroughLayer


def build_layers(builder):
    if builder.verbose:
        print(f"is_triplet={builder.is_triplet}, is_ctl={builder.is_ctl}, alpha={builder.alpha}, beta={builder.beta}")
    anchor_input = layers.Input(name="anchor", shape=builder.input_shape + (builder.channels,))
    positive_input = layers.Input(name="positive", shape=builder.input_shape + (builder.channels,))

    if builder.is_ctl:
        ctl_names = ["positive_ctl", "negative_ctl"] if builder.is_triplet \
            else ["positive_ctl", "negative1_ctl", "negative2_ctl"]
        ctl_input = [layers.Input(name=name, shape=builder.input_shape + (builder.channels,)) for name in ctl_names]
    else:
        ctl_input = None

    if builder.is_triplet:
        negative_inputs = [layers.Input(name="negative", shape=builder.input_shape + (builder.channels,))]
        distance_layer = TripletDistance() if ctl_input is None else TripletCTLDistance()
        loss_layer = TripletLoss(alpha=builder.alpha)
    else:
        negative_inputs = [
            layers.Input(name="negative1", shape=builder.input_shape + (builder.channels,)),
            layers.Input(name="negative2", shape=builder.input_shape + (builder.channels,))
        ]
        distance_layer = QuadrupletDistance() if ctl_input is None else QuadrupletCTLDistance()
        loss_layer = QuadrupletLoss(alpha=builder.alpha, beta=builder.beta)

    if builder.is_ctl:
        if builder.is_triplet:
            input_layers = [anchor_input]
        else:
            input_layers = [anchor_input, negative_inputs[0]]
    else:
        input_layers = [anchor_input, positive_input, *negative_inputs]
    # noinspection PyUnreachableCode
    if builder.preprocess_input:
        encoding_layers = [builder.back_bone(builder.preprocess_input(i)) for i in input_layers]
        if ctl_input:
            encoding_layers = encoding_layers + ctl_input
    else:
        assert False, "Currently just using ResNet which requires Preprocessing"
        encoding_layers = [builder.back_bone(i) for i in input_layers]
        if ctl_input:
            encoding_layers = encoding_layers + ctl_input

    distance_layers = distance_layer(*encoding_layers)

    return input_layers, encoding_layers, distance_layers, loss_layer
