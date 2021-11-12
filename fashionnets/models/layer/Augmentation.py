import tensorflow as tf
from fashionnets.models.layer.RandomHorizontalFlip import RandomHorizontalFlip
from fashionnets.models.layer.RandomErasing import RandomErasing
from tensorflow.keras import layers
import torchvision.transforms as T


def random_crop_layer(crop_shape):
    return layers.Lambda(lambda x: tf.image.random_crop(value=x, size=crop_shape))


def compose_augmentations():
    def __call__(is_train):
        cfg = AugmentationConfig()

        if not is_train:
            return tf.keras.Sequential([
                layers.experimental.preprocessing.Resizing(*cfg.SIZE_TRAIN),
                normalize_image(cfg.PIXEL_MEAN, cfg.PIXEL_STD),
            ])

        pad = cfg.PADDING

        resize_padding_shape = cfg.SIZE_TRAIN[0] + pad, cfg.SIZE_TRAIN[1] + pad

        transform = tf.keras.Sequential([
            layers.experimental.preprocessing.Resizing(*cfg.SIZE_TRAIN),
            RandomHorizontalFlip(p=cfg.PROB),
            resize_with_padding(resize_padding_shape),
            random_crop(cfg.SIZE_TRAIN),
            normalize_image(cfg.PIXEL_MEAN, cfg.PIXEL_STD),
            RandomErasing(probability=cfg.RE_PROB, mean=cfg.PIXEL_MEAN)
        ])

        return transform
    return __call__


# padding, fill=0, padding_mode="constant"

def resize_with_padding(shape):
    return layers.Lambda(lambda x: tf.image.resize_with_pad(
        x, shape[0], shape[1]
    ))




def random_crop(input_shape):
    return layers.Lambda(lambda x:
                         tf.image.random_crop(x, (input_shape[0], input_shape[1], 3))
                         )


import numpy as np
from tensorflow.keras.layers.experimental import preprocessing


def normalize_image(MEAN, STD):
    variance = [np.square(i) for i in STD]

    return preprocessing.Normalization(mean=MEAN, variance=variance)


class AugmentationConfig:
    def __init__(self):
        self.PROB = 0.5  # Probability of Horizontal Flip
        self.RE_PROB = 0.5  # Probability of Random Erasing
        self.SIZE_TRAIN = [224, 224]
        self.PADDING = 10
        self.PIXEL_MEAN = [0.485, 0.456, 0.406]
        self.PIXEL_STD = [0.229, 0.224, 0.225]
