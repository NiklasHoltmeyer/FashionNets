import copy
from collections import defaultdict
from pathlib import Path

from tensorflow import keras
import pandas as pd


class CustomHistoryDump(keras.callbacks.Callback):
    """
    Its just like CSV-Logger. But with Custom Seperator und Save Strategy.
    Some how the normal CSV-Logger does not Append on Google Colab.
    """

    def __init__(self, checkpoint_path, sep=";", decimal_symbol="."):
        super(CustomHistoryDump, self).__init__()
        self.checkpoint_path = checkpoint_path
        self.sep = sep
        self.decimal_symbol = decimal_symbol

    def on_epoch_end(self, epoch, logs=None):
        try:
            self.dumb_history(epoch, logs)
        except:
            pass

    def dumb_history(self, epoch, logs=None):
        logs = logs or defaultdict(lambda: [])

        file_name = f"history_{epoch:04d}.csv"
        csv = Path(self.checkpoint_path, file_name)

        history = self.model.history
        metric_history = history.history
        epochs = history.epoch

        replace_dec = lambda v: str(v).replace(".", ",")
        map_change_decimal_symbol = lambda lst: list(map(replace_dec, lst))
        metric_history_changed_decimal = {k: map_change_decimal_symbol(v) for k, v in metric_history.items()}

        history_csv_data = copy.deepcopy({"epoch": epochs, **metric_history_changed_decimal})
        history_csv_data["epoch"].append(epoch)

        ## add current run
        for k, v in logs.items():
            history_csv_data[k].append(replace_dec(v))


        df = pd.DataFrame(history_csv_data)
        df.to_csv(csv, index=False, sep=";", decimal=self.decimal_symbol)

        print("Epoch", epoch)
        print(logs)
        print("*")
        print(logs.keys())

