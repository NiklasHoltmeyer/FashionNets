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

    @property
    def metrics(self):
        return [self.loss_tracker]

    def save_backbone(self, model_cp_path, epoch):
        backbone_cp_path = Path(model_cp_path, f"backbone-{epoch+1:04d}.ckpt")  # .h5
        backbone_cp_path.parent.mkdir(parents=True, exist_ok=True)

        self.back_bone.save(backbone_cp_path)

