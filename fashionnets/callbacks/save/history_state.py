import copy
from pathlib import Path

from fashiondatasets.utils.logger.defaultLogger import defaultLogger
from tensorflow import keras

from fashionnets.models.states.HistoryState import HistoryState
from tensorflow.keras import backend


class DeepCopyHistory:
    def __init__(self, history):
        self.params = copy.deepcopy(backend.get_value(history.params))
        self.history = copy.deepcopy(backend.get_value(history.history))
        self.epoch = copy.deepcopy(backend.get_value(history.epoch))

    def append_epoch(self, epoch):
        if len(self.epoch) < 1:
            self.epoch = []
        self.epoch.append(epoch)

    def append_logs(self, logs):
        keys = self.history.keys()
        for k, v in logs.items():
            if k not in keys:
                self.history[k] = []
            self.history[k].append(v)


logger = defaultLogger("deepfashion_callbacks")


class SaveHistoryState(keras.callbacks.Callback):
    # noinspection PyUnresolvedReferences
    def __init__(self, checkpoint_path, name):
        super(SaveHistoryState, self).__init__()

        self.model_cp_latest_path = Path(checkpoint_path, name + "_history-")
        self.model_cp_latest_path.parent.mkdir(parents=True, exist_ok=True)

        self.model_cp_latest_path = str(self.model_cp_latest_path.resolve())

    #    def on_epoch_begin(self, epoch, logs=None):
    #        # on_epoch_end does not include latest Results!
    #        # Just overwrite latest State to Contain All Predictions from last Run!
    #        if epoch == 0:
    #            return

    #        self.save_state((int(epoch) - 1), force=True)  # Overwrite
    #        print("Epoch", epoch)
    #        print("Epoch - 1", (epoch - 1))
    #        print("Epoch(i) - 1", (int(epoch) - 1))

    def on_epoch_end(self, epoch, logs=None):
        history_copy = DeepCopyHistory(self.model.history)
        history_copy.append_epoch(epoch)
        history_copy.append_logs(logs)

        self.save_state(epoch, history_copy)

    def save_state(self, epoch, history_copy):
        try:
            cp_path = self.model_cp_latest_path + f"{epoch:04d}.pkl"
            path = str(Path(cp_path).resolve())

            HistoryState(history_copy).save(path)

        except Exception as e:
            logger.error("HistoryState::save_state")  # easier to trace Exception from withing Google Colab
            raise Exception(e)
