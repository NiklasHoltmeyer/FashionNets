from pathlib import Path

from tensorflow import keras


class CustomSaveModel(keras.callbacks.Callback):
    def __init__(self, checkpoint_path, name, save_format=None):
        super(CustomSaveModel, self).__init__()
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
        self.model.save(fPath, save_format=self.save_format)


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
