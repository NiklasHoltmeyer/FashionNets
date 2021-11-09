from pathlib import Path

from tensorflow import keras


class SaveModel(keras.callbacks.Callback):
    def __init__(self, checkpoint_path, name, save_format=None):
        super(SaveModel, self).__init__()
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