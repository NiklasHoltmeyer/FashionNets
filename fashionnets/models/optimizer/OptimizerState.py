import pickle
import tensorflow as tf
from tensorflow.keras import backend

class OptimizerState:
    def __init__(self, optimizer):
        self.config = optimizer.get_config()
        self.weights = optimizer.get_weights()
#        self.iterations = optimizer.iterations

        self.lr = backend.get_value(optimizer.lr)
        self.iterations = backend.get_value(optimizer.iterations)
        self.learning_rate = backend.get_value(optimizer.learning_rate)

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

    def apply_weights(self, model, optimizer):
        grad_vars = model.trainable_weights

        zero_grads = [tf.zeros_like(w) for w in grad_vars]
        optimizer.apply_gradients(zip(zero_grads, grad_vars))

        optimizer.set_weights(self.weights)

    def apply(self, model):
        optimizer = self.empty_optimizer()
        self.apply_weights(model, optimizer)

        backend.set_value(self.lr)
        backend.set_value(self.iterations)
        backend.set_value(self.learning_rate)

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

    def __str__(self):
        lines = ["{"]
        for k, v in self.__dict__.items():
            if k == "weights":
                line = f"\t{k}:\t{str(type(v))},"
            else:
                line = f"\t{k}:\t{v},"
            lines.append(line)
        lines.append("}")
        return "\n".join(lines)
