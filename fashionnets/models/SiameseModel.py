import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import metrics
from tensorflow.keras import optimizers

from fashionnets.models.embedding.resnet50 import ResNet50Builder
from fashionnets.networks.SiameseNetwork import SiameseNetwork


class SiameseModel(Model):
    # https://keras.io/examples/vision/siamese_network/
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, triplet_loss, alpha, beta=None):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network

        self.alpha = alpha
        self.beta = beta

        self.loss_tracker = metrics.Mean(name="loss")
        if triplet_loss:
            self._compute_loss = self._compute_triplet_loss_
        else:
            self._compute_loss = self._compute_quadruplet_loss_
            assert beta, "Beta must be set."

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_triplet_loss_(self, data):  # Triplet loss = AP - AN + alpha
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance + self.alpha
        loss = tf.maximum(loss, 0.0)
        return loss

    def _compute_quadruplet_loss_(self, data):  # AP-AN + alpha_1 + AP-NN + alpha_2
        ap_distance, an_distance, nn_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = tf.maximum((ap_distance - an_distance + self.alpha), 0) + \
               tf.maximum((ap_distance - nn_distance + self.beta), 0)

        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]


if __name__ == "__main__":
    target_shape = (144, 144)

    embedding, preprocess_input = ResNet50Builder.build(target_shape)
    siamese_network = SiameseNetwork.build(embedding, triplets=False, input_shape=target_shape,
                                           preprocess_input=preprocess_input,
                                           channels=3)

    siamese_model = SiameseModel(siamese_network, triplet_loss=True)
    siamese_model.compile(optimizer=optimizers.Adam(0.0001))
    # alpha 1, beta 0.5
