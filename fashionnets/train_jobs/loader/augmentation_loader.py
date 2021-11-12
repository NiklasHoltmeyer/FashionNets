import tensorflow as tf
from fashionnets.models.layer.Augmentation import random_crop_layer
from fashionnets.models.layer.RandomErasing import RandomErasing
from tensorflow.keras import layers

#def load_augmentations(**train_job):
#    augmentation_description = train_job["augmentation_description"]
#    crop_shape = train_job["input_shape"]

#    augmentations = {
        #"rrc": random_crop_layer(train_job["input_shape"]),
        #"h_flip": tf.keras.layers.RandomFlip("horizontal"),
        #"re": RandomErasing()
#    }

#    full = tf.keras.Sequential([]
#    )
#    return {
        #"full":
#    }[augmentation_description]


#def random_crop(input_shape):
#    resize_and_rescale = tf.keras.Sequential([
        #random_crop_layer(input_shape),
#    ])

#data_augmentation = tf.keras.Sequential([
#  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
#  layers.experimental.preprocessing.RandomRotation(0.2),
#])