import pickle
from pathlib import Path

from tensorflow import keras


class SaveHistory(keras.callbacks.Callback):
    def __init__(self, checkpoint_path, name):
        super(SaveHistory, self).__init__()

        self.model_cp_latest_path = Path(checkpoint_path, name + "_history-")
        self.model_cp_latest_path.parent.mkdir(parents=True, exist_ok=True)

        self.model_cp_latest_path = str(self.model_cp_latest_path.resolve())

    def on_epoch_end(self, epoch, logs=None):
        cp_path = self.model_cp_latest_path + f"{epoch:04d}.pkl"
        path = str(Path(cp_path).resolve())
        try:
            with open(path, 'wb') as f:
                pickle.dump(self.model.history_history, f)
        except Exception as e:
            print("Could not Save History:")
            print(e)
