from pathlib import Path

import tensorflow as tf
from fashiondatasets.utils.logger.defaultLogger import defaultLogger
from tensorflow.keras import Model
from tensorflow.keras import metrics

from fashionnets.models.embedding.resnet50 import EMBEDDING_DIM


# noinspection PyAbstractClass,PyMethodOverriding,PyAbstractClass
class SiameseModel(Model):
    # https://keras.io/examples/vision/siamese_network/
    def __init__(self, siamese_network, back_bone):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.loss_tracker = metrics.Mean(name="loss")
        self.logger = defaultLogger(name="Siamese_Model")
        self.back_bone = back_bone

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self.siamese_network(data)

        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self.siamese_network(data)

        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}

    def fake_predict(self):
        """
        Force Init Layers (-> Model cant be Saved by Training without Init. Layers)
        """

        input_shape = self.siamese_network.input_shape_
        is_triplet = self.siamese_network.is_triplet
        is_ctl = self.siamese_network.is_ctl

        if is_ctl:
            random_data = (1,) + input_shape + (3,)
            random_a = tf.random.uniform(random_data)

            embedding_shape = (1, EMBEDDING_DIM)
            # random_centroid = tf.random.uniform(embedding_shape)

            random_centroid = [tf.random.uniform(embedding_shape)] * (2 if is_triplet else 3)
            # random_centroid = [random_centroid] * (2 if is_triplet else 3)
            data = [random_a, *random_centroid]
        else:
            image_shape = (1,) + input_shape + (3,)
            random_apn = tf.random.uniform(image_shape)
            random_apn = [random_apn] * (3 if is_triplet else 4)
            data = random_apn
        return self.predict(data)

    def validate_embedding(self, small_batch):
        """
        Check if Embeddings are Constant -> bad
        :return:
        """

        def is_embedding_constant():
            #            random_data = (1,) + input_shape + (3,)
            #            random_ds = [tf.random.uniform(random_data)] * (3 if is_triplet else 4)

            test_embeddings = self.siamese_network.embed(small_batch)

            is_constant = lambda a, b: tf.math.reduce_sum(tf.math.square(a - b)) == 0

            for i in range(len(test_embeddings)):
                j = (i + 1) % len(test_embeddings)

                assert i != j
                x, y = test_embeddings[i], test_embeddings[j]
                x_nans = tf.reduce_any(tf.math.is_nan(x))
                y_nans = tf.reduce_any(tf.math.is_nan(y))

                if x_nans or y_nans:
                    raise Exception("Embedding Space contains NaN's!.")

                _const = is_constant(x, y)
                if _const is not False:
                    return False
            return True

        if is_embedding_constant():
            raise Exception("The Embedding-Model seems to produce constant results.")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def save_backbone(self, model_cp_path, epoch):
        backbone_cp_path = Path(model_cp_path, f"backbone-{epoch:04d}.ckpt")
        backbone_cp_path.parent.mkdir(parents=True, exist_ok=True)

        self.back_bone.save(backbone_cp_path)

    def load_embedding_weights(self, cp_path):
        if not Path(cp_path).exists():
            raise Exception(f"Checkpoint Path does not Exist! {cp_path}")
        self.back_bone.load_weights(cp_path)

    def extract_features(self, images):
        return self.siamese_network.extract_features(images)
