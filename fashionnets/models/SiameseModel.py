import tensorflow as tf
from fashiondatasets.utils.logger.defaultLogger import defaultLogger
from tensorflow.keras import Model
from tensorflow.keras import metrics
from tensorflow.keras import optimizers

from fashionnets.models.embedding.resnet50 import ResNet50Builder
from fashionnets.networks.SiameseNetwork import SiameseNetwork


class SiameseModel(Model):
    # https://keras.io/examples/vision/siamese_network/
    def __init__(self, siamese_network):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.loss_tracker = metrics.Mean(name="loss")
        self.logger = defaultLogger(name="Siamese_Modell")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        self.logger.debug("Train_Step: Calc Loss")
        with tf.GradientTape() as tape:
            loss = self.siamese_network(data)
        self.logger.debug("Train_Step: Tape.Gradient")
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)
        self.logger.debug("Train_Step: Gradient.Apply")
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )
        self.logger.debug("Train_Step: Update Loss")
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        self.logger.debug("Test_Step: Calc Val_Loss")
        loss = self.siamese_network(data)

        self.loss_tracker.update_state(loss)
        self.logger.debug("Test_Step: Update Val_Loss")
        return {"loss": self.loss_tracker.result()}

    def fake_predict(self, input_shape, is_triplet):
        """
        Force Init Layers (-> Model cant be Saved by Training without Init. Layers)
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
