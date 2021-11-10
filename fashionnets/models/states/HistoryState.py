import collections
import os
import pickle
from collections import defaultdict

from fashionscrapper.utils.list import distinct
from tensorflow.keras import backend


class HistoryState:
    def __init__(self, history, params=None, history_history=None, epoch=None):
        if history is not None:
            self.params = backend.get_value(history.params)
            self.history_history = backend.get_value(history.history)
            self.epoch = backend.get_value(history.epoch)
        else:
            self.params = params
            self.history_history = history_history
            self.epoch = epoch

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        if not path.endswith(".pkl"):
            params, epochs, history = HistoryState.load_histories(path)
            return HistoryState(history=None, params=params, history_history=history, epoch=epochs)
        with open(path, 'rb') as f:
            state = pickle.load(f)
            return state

    @staticmethod
    def load_histories(path):
        files = os.listdir(path)
        history_file_paths = filter(lambda f: "history" in f, files)
        history_file_paths = filter(lambda f: f.endswith(".pkl"), history_file_paths)
        history_file_paths = map(lambda f: os.path.join(path, f), history_file_paths)

        history_files = map(HistoryState.load, history_file_paths)
        history_files = list(history_files)

        if len(history_files) < 1:
            raise Exception("Could not Find History File! (File must Contain 'history' and End with '.pkl')")

        if len(history_files) == 1:
            return history_files[0]

        history_files = sorted(history_files, key=lambda o: o.epoch[-1])
        history_files = list(history_files)

        latest_history = history_files[-1]

        params = latest_history.params
        values_by_ep_metric = defaultdict(lambda: defaultdict(lambda: []))
        epoch_values = [(h.epoch, h.history_history) for h in history_files]

        for epoch, values in epoch_values:
            for metric, metric_values in values.items():
                for _epoch, value in zip(epoch, metric_values):
                    values_by_ep_metric[_epoch][metric].append(value)

        values_by_ep_metric = dict(collections.OrderedDict(sorted(values_by_ep_metric.items())))
        values_by_ep_metric = {k: dict(v) for k, v in values_by_ep_metric.items()}

        epochs = []
        history = defaultdict(lambda: [])
        for ep, values in values_by_ep_metric.items():
            epochs.append(ep)
            for metric, metric_values in values.items():
                metric_values = distinct(metric_values)
                assert len(metric_values) == 1
                metric_value = metric_values[0]

                history[metric].append(metric_value)
        number_elements = distinct([len(x) for x in history.values()])
        assert len(number_elements) == 1, "ALl Metric Values should have the same Number of Values!"
        assert number_elements[0] == len(epochs)

        return params, epochs, dict(history)

    def apply(self, model):
        model.history.params = self.params
        model.history.history = self.history_history
        model.history.epoch = self.epoch

    def __eq__(self, other):
        if not isinstance(other, HistoryState):
            return False

        for attribute in self.__dict__.keys():
            if getattr(self, attribute) != getattr(other, attribute):
                return False

        return True

    def __str__(self):
        lines = ["{"]
        for k, v in self.__dict__.items():
            line = f"\t{k}:\t{v},"
            lines.append(line)
        lines.append("}")
        return "\n".join(lines)


if __name__ == "__main__":
    # path = r"D:\Download\311_resnet50_imagenet_quadruplet_history-0004.pkl"
    # print(HistoryState.load(path))
    path = r"D:\Download\312_resnet50_imagenet_triplet0009"
    print(HistoryState.load(path))
