import pickle
from pathlib import Path

from tensorflow import keras
from tensorflow.keras import backend as K

assert K is not None or True

class CustomSaveModel(keras.callbacks.Callback):
    def __init__(self, checkpoint_path, name, save_format=None):
        super(CustomSaveModel, self).__init__()
        assert save_format in ["tf", "h5", "ckpt", None]

        self.model_cp_latest_path = Path(checkpoint_path, name + "_cp-")
        self.model_cp_latest_path.parent.mkdir(parents=True, exist_ok=True)

        self.model_cp_latest_path = str(self.model_cp_latest_path.resolve())

        self.save_format = save_format
        self.ext = f".{save_format}" if save_format else ""

    def on_epoch_end(self, epoch, logs=None):
        cp_path = self.model_cp_latest_path + f"{epoch:04d}" + self.ext
        path = str(Path(cp_path).resolve())
        self.model.save(path, save_format=self.save_format)


class CustomSaveOptimizerState(keras.callbacks.Callback):
    def __init__(self, checkpoint_path, name):
        super(CustomSaveOptimizerState, self).__init__()

        self.model_cp_latest_path = Path(checkpoint_path, name + "_opt-")
        self.model_cp_latest_path.parent.mkdir(parents=True, exist_ok=True)

        self.model_cp_latest_path = str(self.model_cp_latest_path.resolve())

    def on_epoch_end(self, epoch, logs=None):
        cp_path = self.model_cp_latest_path + f"{epoch:04d}.pkl"
        path = str(Path(cp_path).resolve())

        symbolic_weights = getattr(self.model.optimizer, 'weights')
        weight_values = K.batch_get_value(symbolic_weights)
        with open(path, 'wb') as f:
            pickle.dump(weight_values, f)


class CustomSaveWeights(keras.callbacks.Callback):
    def __init__(self, checkpoint_path, name, save_format=None):
        super(CustomSaveWeights, self).__init__()
        assert save_format in ["tf", "h5", None]

        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.mkdir(exist_ok=True, parents=True)

        self.checkpoint_path = str(checkpoint_path.resolve())

        self.name = name
        self.save_format = save_format
        self.ext = f".{save_format}" if save_format else ""

    def on_epoch_end(self, epoch, logs=None):
        fName = f"/{self.name}_ep{epoch}{self.ext}"
        fPath = Path(self.checkpoint_path + fName)
        self.model.save_weights(fPath, save_format=self.save_format)
