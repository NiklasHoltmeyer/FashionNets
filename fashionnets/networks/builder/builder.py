from fashionnets.metrics.distance_layers import TripletDistance, QuadrupletDistance, TripletCTLDistance, \
    QuadrupletCTLDistance
from fashionnets.metrics.loss_layer import QuadrupletLoss, TripletLoss
from fashionnets.models.embedding.resnet50 import EMBEDDING_DIM
from fashionnets.models.embedding.simple_cnn import SimpleCNN
from tensorflow.keras import layers


def build_layers(builder):
    if builder.verbose:
        print(f"is_triplet={builder.is_triplet}, is_ctl={builder.is_ctl}, alpha={builder.alpha}, beta={builder.beta}")
    anchor_input = layers.Input(name="anchor", shape=builder.input_shape + (builder.channels,))
    positive_input = layers.Input(name="positive", shape=builder.input_shape + (builder.channels,))

    if builder.is_ctl:
        ctl_names = ["positive_ctl", "negative_ctl"] if builder.is_triplet \
            else ["positive_ctl", "negative1_ctl", "negative2_ctl"]
        ctl_input = [layers.Input(name=name, shape=EMBEDDING_DIM) for name in ctl_names]
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
            input_layers = input_layers + ctl_input
    else:
        assert False, "Currently just using ResNet which requires Preprocessing"
        encoding_layers = [builder.back_bone(i) for i in input_layers]
        if ctl_input:
            encoding_layers = encoding_layers + ctl_input
            input_layers = input_layers + ctl_input

    distance_layers = distance_layer(*encoding_layers)

    return input_layers, encoding_layers, distance_layers, loss_layer

if __name__ == "__main__":
    from tensorflow.keras import Model

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

    mock_sn = FakeBuilder()
    input_layers, encoding_layers, distance_layers, loss_layer = build_layers(builder=mock_sn)
    output_layers = loss_layer(*encoding_layers)
    for l in input_layers:
        print(l)
    exit(0)
    full_model = Model(
        inputs=input_layers, outputs=output_layers
    )

    print(full_model.summary())

    embedding_model = Model(inputs=input_layers, outputs=encoding_layers)
    input_layer = layers.Input(name="input_image", shape=mock_sn.input_shape + (mock_sn.channels,))
    encoding = mock_sn.back_bone(mock_sn.preprocess_input(input_layer))
    feature_extractor = Model(inputs=[input_layer], outputs=encoding)
