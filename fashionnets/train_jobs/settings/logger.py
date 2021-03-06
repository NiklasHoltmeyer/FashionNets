import logging

from fashiondatasets.utils.logger.defaultLogger import defaultLogger

logger_names = [
    "deepfashion_callbacks",
    "deepfashion_model_builder",
    "deepfashion_data_builder",
    "deepfashion_environment",
    "FashionNet"
]

verbose_logger = [
    "fashion_pair_gen",
    "deepfashion_data_builder",
    "FashionNet"
]

for l in verbose_logger:
    defaultLogger(l).setLevel(logging.DEBUG)






