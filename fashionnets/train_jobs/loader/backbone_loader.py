from fashionnets.models.embedding.resnet50 import ResNet50Builder
from fashionnets.models.embedding.simple_cnn import SimpleCNN
from fashionnets.train_jobs.jobs import format_name

def load_backbone_info_resnet(input_shape, back_bone, is_triplet, weights="imagenet"):
    _format = format_name(is_triplet)
    run_name = f"{back_bone}_{weights}_{_format}"

    back_bone, preprocess_input = ResNet50Builder.build(input_shape=input_shape, weights=weights)

    return run_name, back_bone, preprocess_input


def load_backbone_info_simple_cnn(input_shape, back_bone, is_triplet):
    weights = "none"
    _format = format_name(is_triplet)
    run_name = f"{back_bone}_{weights}_{_format}"

    back_bone, preprocess_input = SimpleCNN.build(input_shape)  # lambda i: i / 255.

    return run_name, back_bone, preprocess_input
