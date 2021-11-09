from collections import defaultdict
from pathlib import Path

import pandas as pd
from tensorflow import keras


class CSVLogger(keras.callbacks.Callback):
    """
    Its just like CSV-Logger. But with Custom Separator und Save Strategy.
    Some how the normal CSV-Logger does not Append on Google Colab.
    """

    def __init__(self, checkpoint_path, sep=";", decimal_symbol="."):
        super(CSVLogger, self).__init__()
        self.checkpoint_path = checkpoint_path
        self.sep = sep
        self.decimal_symbol = decimal_symbol

    def on_epoch_end(self, epoch, logs=None):
        try:
            self.dumb_history(epoch, logs)
        except Exception as e:
            print("CustomHistoryDump::on_epoch_end")
            raise e

    def dumb_history(self, epoch, logs=None):
        if epoch < 1:
            return

        logs = logs or defaultdict(lambda: [])

        file_name = f"history_{epoch:04d}.csv"
        csv = Path(self.checkpoint_path, file_name)

        history = self.model.history_history
        metric_history = history.history_history
        epochs = history.epoch

        replace_dec = lambda v: str(v).replace(".", ",")
        map_change_decimal_symbol = lambda lst: list(map(replace_dec, lst))
        metric_history_changed_decimal = {k: map_change_decimal_symbol(v) for k, v in metric_history.items()}

        history_csv_data = {"epoch": epochs + [epoch], **metric_history_changed_decimal}

        # add current run

        # noinspection PyBroadException
        try:
            for k, v in logs.items():
                history_csv_data[k].append(replace_dec(v))

            df = pd.DataFrame(history_csv_data)
            df.to_csv(csv, index=False, sep=";", decimal=self.decimal_symbol)
        except:
            pass
