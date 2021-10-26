from fashionnets.models.embedding.resnet50 import ResNet50Builder
from fashionnets.models.embedding.simple_cnn import SimpleCNN


def load_backbone_info_resnet(input_shape, back_bone_name, is_triplet, weights):  # =weights="imagenet"
    run_name = back_bone_configuration_name(back_bone_name, is_triplet, weights)

    back_bone, preprocess_input = ResNet50Builder.build(input_shape=input_shape, weights=weights)

    return run_name, back_bone, preprocess_input


def load_backbone_info_simple_cnn(input_shape, back_bone_name, is_triplet, weights=None):
    assert weights is None

    run_name = back_bone_configuration_name(back_bone_name, is_triplet, weights)

    back_bone, preprocess_input = SimpleCNN.build(input_shape)  # lambda i: i / 255.

    return run_name, back_bone, preprocess_input


def back_bone_configuration_name(back_bone, is_triplet, weights):
    _format = format_name(is_triplet)
    run_name = f"{back_bone}_{weights}_{_format}"
    return run_name


def format_name(is_triplet):
    return "triplet" if is_triplet else "quadruplet"