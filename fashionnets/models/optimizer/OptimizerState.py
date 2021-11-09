import pickle
import tensorflow as tf


class OptimizerState:
    def __init__(self, optimizer):
        self.config = optimizer.get_config()
        self.weights = optimizer.get_weights()
        self.iterations = optimizer.iterations
        self.variables = optimizer.variables()

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            state = pickle.load(f)
            return state

    def empty_optimizer(self):
        if self.config["name"] == "Adam":
            return tf.keras.optimizers.Adam(
                **self.config
            )
        raise Exception(f"Optimizer {self.config['name']} is not Supported!")

    def apply(self, model):
        optimizer = self.empty_optimizer()
        optimizer.weights.extend(self.weights)
        optimizer.iterations = self.iterations
        optimizer.variables = self.variables

        model.optimizer = optimizer

    def __eq__(self, other):
        if not isinstance(other, OptimizerState):
            return False
        for attribute in self.__dict__.keys():
            if attribute == "weights":
                continue
            if getattr(self, attribute) != getattr(other, attribute):
                return False

            if len(self.weights) != len(other.weights):
                return False

            # weights_equal = tf.math.reduce_all(  #<- memory issues on colab
            # [tf.math.reduce_all(a == b) for a, b in zip(self.weights, other.weights)]
            # )
            weights_equal = all(
                map(lambda xy: tf.math.reduce_all(xy[0] == xy[1]).numpy(), zip(self.weights, other.weights)))

            return weights_equal
