from fashionnets.models.embedding.resnet50 import ResNet50Builder
from fashionnets.models.embedding.simple_cnn import SimpleCNN


def job_resnet(input_shape, back_bone, is_triplet, weights="mobile_net"):
    _format = "triplet" if is_triplet else "quadruplet"
    run_name = f"{back_bone}_{weights}_{_format}"

    back_bone, preprocess_input = ResNet50Builder.build(input_shape)
    if not weights:
        preprocess_input = None

    return run_name, back_bone, preprocess_input


def job_simple_cnn(input_shape, back_bone, is_triplet):
    weights = "none"
    _format = "triplet" if is_triplet else "quadruplet"
    run_name = f"{back_bone}_{weights}_{_format}"

    back_bone, preprocess_input = SimpleCNN.build(input_shape), None

    return run_name, back_bone, preprocess_input