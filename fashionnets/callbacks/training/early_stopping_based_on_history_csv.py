from fashiondatasets.utils.logger.defaultLogger import defaultLogger
from tensorflow import keras

from fashionnets.util.csv import HistoryCSVHelper

logger = defaultLogger("deepfashion_callbacks")

class EarlyStoppingBasedOnHistory(keras.callbacks.Callback):
    def __init__(self, history_path, monitor="loss", patience=3, sep=",", lower_is_better=True):
        super(EarlyStoppingBasedOnHistory, self).__init__()
        self.history_path = history_path
        self.sep = sep
        self.patience = patience
        self.monitor = monitor

        self.comparative_operator = min if lower_is_better else max

    def on_epoch_begin(self, epoch, logs=None):
        history = HistoryCSVHelper.history_csv_to_dict(self.history_path, sep=self.sep)  # drop_columns=["epoch"]

        if len(history) < 1 or epoch < self.patience:
            return

        ep_values = zip(history["epoch"], history[self.monitor])
        best_epoch, best_value = self.comparative_operator(ep_values, key=lambda d: d[1])

        early_stopping = (best_epoch + self.patience) < epoch
        if early_stopping:
            logger.debug(f"Best {self.monitor} Results at {best_epoch}.")
            logger.debug(f"Stop Training. {self.monitor} has not improved in {epoch - best_epoch} Epochs!")
            self.model.stop_training = True

#        if len(history) < 1:
#            return

#        for metric, values in history.items():
#            stop, best_ep = self.early_stopping(values, epoch)
#            if stop:
#                print(f"Best {metric} Results at {best_ep}")
#                if metric == self.monitor:
#                    print(f"Stop Training. {metric} has not improved in {epoch - best_ep} Epochs!")
#                    self.model.stop_training = True

#    def early_stopping(self, values, epoch):
#        if len(values) < 1:
#            return False, -1

#        best_epoch = values.index(min(values))

#        if epoch < self.patience:  # Epoch starts at 0. so it trains at-least for patience epochs
#            return False, best_epoch

#        return (best_epoch + self.patience) < epoch, best_epoch

if __name__ == "__main__":
    path = r"F:\workspace\FashNets\1337_resnet50_None_triplet\history.csv"
    csv_sep = ";"
    EarlyStoppingBasedOnHistory(path, monitor='loss', patience=3, sep=csv_sep).on_epoch_begin(9, None)
