from tensorflow import keras
from fashionnets.util.csv import HistoryCSVHelper


class EarlyStoppingBasedOnHistory(keras.callbacks.Callback):
    def __init__(self, history_path, monitor="loss", patience=3, sep=","):
        super(EarlyStoppingBasedOnHistory, self).__init__()
        self.history_path = history_path
        self.sep = sep
        self.patience = patience
        self.monitor = monitor

    def on_epoch_begin(self, epoch, logs=None):
        history = HistoryCSVHelper.history_csv_to_dict(self.history_path, drop_columns=["epoch"], sep=self.sep)

        if len(history) < 1:
            return

        for metric, values in history.items():
            stop, best_ep = self.early_stopping(values, epoch)
            if stop:
                print(f"Best {metric} Results at {best_ep}")
                if metric == self.monitor:
                    print(f"Stop Training. {metric} has not improved in {epoch - best_ep} Epochs!")
                    self.model.stop_training = True

    def early_stopping(self, values, epoch):
        if len(values) < 1:
            return False, -1

        best_epoch = values.index(min(values))

        if epoch < self.patience: # Epoch starts at 0. so it trains atleast for patience epochs
            return False, best_epoch

        return (best_epoch + self.patience) < epoch, best_epoch
