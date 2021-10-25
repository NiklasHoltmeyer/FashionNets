from fashiondatasets.consts import QUADRUPLET_KEY, TRIPLET_KEY
from fashionnets.models.embedding.resnet50 import ResNet50Builder
from fashionnets.models.embedding.simple_cnn import SimpleCNN


def load_backbone_info_resnet(input_shape, back_bone, is_triplet, weights="imagenet"):
    _format = TRIPLET_KEY if is_triplet else QUADRUPLET_KEY
    run_name = f"{back_bone}_{weights}_{_format}"

    back_bone, preprocess_input = ResNet50Builder.build(input_shape=input_shape, weights=weights)

    return run_name, back_bone, preprocess_input


def load_backbone_info_simple_cnn(input_shape, back_bone, is_triplet):
    weights = "none"
    _format = TRIPLET_KEY if is_triplet else QUADRUPLET_KEY
    run_name = f"{back_bone}_{weights}_{_format}"

    back_bone, preprocess_input = SimpleCNN.build(input_shape)  # lambda i: i / 255.

    return run_name, back_bone, preprocess_input
