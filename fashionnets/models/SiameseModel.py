from pathlib import Path

import tensorflow as tf
from fashiondatasets.utils.logger.defaultLogger import defaultLogger
from tensorflow.keras import Model
from tensorflow.keras import metrics


# noinspection PyAbstractClass,PyMethodOverriding
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
            loss = self.siamese_network(data) * 1000

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

    def fake_predict(self, input_shape, is_triplet):
        """
        Force Init Layers (-> Model cant be Saved by Training without Init. Layers)
        :param is_triplet:
        :param input_shape:
        :return:
        """
        random_data = (1,) + input_shape + (3,)
        random_ds = tf.random.uniform(random_data)
        random_ds = [random_ds] * (3 if is_triplet else 4)
        self.predict(random_ds)

    def validate_embedding(self, small_batch):
        """
        Check if Embeddings are Constant -> bad
        :return:
        """

        def is_embedding_constant():
#            random_data = (1,) + input_shape + (3,)
#            random_ds = [tf.random.uniform(random_data)] * (3 if is_triplet else 4)

            test_emmbeddings = self.siamese_network.embed(small_batch)

            is_constant = lambda x, y: tf.math.reduce_sum(tf.math.square(x - y)) == 0

            for i in range(len(test_emmbeddings)):
                j = (i + 1) % len(test_emmbeddings)

                assert i != j
                x, y = test_emmbeddings[i], test_emmbeddings[j]
                x_nans = tf.reduce_any(tf.math.is_nan(x))
                y_nans = tf.reduce_any(tf.math.is_nan(y))

                if x_nans or y_nans:
                    raise Exception("Embedding Space contains NaN's!.")

                _const = is_constant(x, y)
                if _const is not False:
                    return False
            return True

        if is_embedding_constant:
            raise Exception("The Embedding-Model seems to produce constant results.")


    @property
    def metrics(self):
        return [self.loss_tracker]

    def save_backbone(self, model_cp_path, epoch):
        backbone_cp_path = Path(model_cp_path, f"backbone-{epoch+1:04d}.ckpt")  # .h5
        backbone_cp_path.parent.mkdir(parents=True, exist_ok=True)

        self.back_bone.save(backbone_cp_path)

