from pathlib import Path
import pandas as pd


class HistoryCSVHelper:
    @staticmethod
    def history_csv_to_dict(history_path, drop_columns=None, sep=","):
        if not Path(history_path).exists():
            return {}

        try:
            history_df = pd.read_csv(history_path, sep=sep)
        except pd.errors.EmptyDataError:
            return {}

        if history_df.empty:
            return {}

        if drop_columns:
            if not type(drop_columns) == list:
                drop_columns = [drop_columns]

            for column in drop_columns:
                history_df.drop(column, inplace=True, axis=1)

        history_dict = {}

        for key in history_df.keys():
            values = list(history_df[key].values)
            history_dict[key] = values

        return history_dict

    @staticmethod
    def last_epoch(history_path, sep):
        history_dict = HistoryCSVHelper.history_csv_to_dict(history_path, sep=sep)
        epoch_info = history_dict.get("epoch", [-1])
        last_epoch = epoch_info[-1]
        return last_epoch

    @staticmethod
    def last_epoch_from_train_job(train_job):
        history_path = Path(train_job["path"]["checkpoint"], "history.csv")
        if history_path.exists():
            sep = ";"
            return HistoryCSVHelper.last_epoch(history_path, sep)
        return None
