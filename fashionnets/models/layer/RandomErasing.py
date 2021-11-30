import tensorflow as tf
from tensorflow.keras import layers


class RandomErasing(layers.Layer):
    # SRC https://github.com/Janghyeonwoong/Random-Erasing-Tensorflow2/blob/master/erasing.py Adapted as Layer

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465), method="mean",
                 **kwargs):

        # im Paper sl = 0.02, r1 = 1/r2, sh = 0.4, r1=0.3
        super().__init__(**kwargs)
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

        assert method in ['random', 'white', 'black', "mean"], 'Wrong method parameter'

        if method is 'black':
            self.part_3_method = lambda h, w, img_channels: tf.zeros((h, w, img_channels), dtype=tf.float32)
        elif method is 'white':
            self.part_3_method = lambda h, w, img_channels: tf.ones((h, w, img_channels), dtype=tf.float32)
        elif method is 'random':
            self.part_3_method = lambda h, w, img_channels: tf.random.uniform((h, w, img_channels), dtype=tf.float32)
        elif method is "mean":
            mean_mean = sum(mean) / len(mean)
            self.part_3_method = lambda h, w, img_channels: tf.zeros((h, w, img_channels), dtype=tf.float32) + mean_mean

    def call(self, img):
        # Motivated by https://github.com/Amitayus/Random-Erasing-TensorFlow.git
        # Motivated by https://github.com/zhunzhong07/Random-Erasing/blob/master/transforms.py
        """
            Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
            -------------------------------------------------------------------------------------
            img : 3D Tensor data (H,W,Channels) normalized value [0,1]
            probability: The probability that the operation will be performed.
            sl: min erasing area
            sh: max erasing area
            r1: min aspect ratio
            method : 'black', 'white' or 'random'. Erasing type
            -------------------------------------------------------------------------------------
        """

        if tf.random.uniform([]) > self.probability:
            return img

        img_width = img.shape[1]
        img_height = img.shape[0]
        img_channels = img.shape[2]

        area = img_height * img_width

        h, w = img_height + 10, img_height + 10  # just force  h > img_height and w > img_height

        while tf.constant(True, dtype=tf.bool):
            if h > img_height or w > img_height or h is None or w is None:
                target_area = tf.random.uniform([], minval=self.sl, maxval=self.sh) * area
                aspect_ratio = tf.random.uniform([], minval=self.r1, maxval=1 / self.r1)
                h = tf.cast(tf.math.round(tf.math.sqrt(target_area * aspect_ratio)), tf.int32)
                w = tf.cast(tf.math.round(tf.math.sqrt(target_area / aspect_ratio)), tf.int32)
            else:
                break

        x1 = tf.cond(img_height == h, lambda: 0,
                     lambda: tf.random.uniform([], minval=0, maxval=img_height - h, dtype=tf.int32))
        y1 = tf.cond(img_width == w, lambda: 0,
                     lambda: tf.random.uniform([], minval=0, maxval=img_width - w, dtype=tf.int32))

        part1 = tf.slice(img, [0, 0, 0], [x1, img_width, img_channels])  # first row
        part2 = tf.slice(img, [x1, 0, 0], [h, y1, img_channels])  # second row 1

        part3 = self.part_3_method(h, w, img_channels)

        part4 = tf.slice(img, [x1, y1 + w, 0], [h, img_width - y1 - w, img_channels])  # second row 3
        part5 = tf.slice(img, [x1 + h, 0, 0], [img_height - x1 - h, img_width, img_channels])  # third row

        middle_row = tf.concat([part2, part3, part4], axis=1)
        img = tf.concat([part1, middle_row, part5], axis=0)

        return img
