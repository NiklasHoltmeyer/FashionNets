from pathlib import Path

from tensorflow import keras

from fashionnets.models.states.HistoryState import HistoryState
from fashionnets.models.states.OptimizerState import OptimizerState


class HistoryState(keras.callbacks.Callback):
    def __init__(self, checkpoint_path, name):
        super(HistoryState, self).__init__()

        self.model_cp_latest_path = Path(checkpoint_path, name + "_history-")
        self.model_cp_latest_path.parent.mkdir(parents=True, exist_ok=True)

        self.model_cp_latest_path = str(self.model_cp_latest_path.resolve())

    def on_epoch_end(self, epoch, logs=None):
        try:
            cp_path = self.model_cp_latest_path + f"{epoch:04d}.pkl"
            path = str(Path(cp_path).resolve())

            HistoryState(self.model.history).save(path)

        except Exception as e:
            print("CustomSaveOptimizerState::on_epoch_end")  # easier to trace Exception from withing Google Colab
            raise Exception(e)


