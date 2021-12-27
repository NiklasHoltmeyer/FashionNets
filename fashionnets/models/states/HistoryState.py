import collections
import os
import pickle
from collections import defaultdict

from fashionscrapper.utils.list import distinct, flatten
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
    def load(path, skip_assert=False):
        if not path.endswith(".pkl"):
            params, epochs, history = HistoryState.load_histories(path, skip_assert=skip_assert)
            return HistoryState(history=None, params=params, history_history=history, epoch=epochs)
        with open(path, 'rb') as f:
            state = pickle.load(f)
            return state

    @staticmethod
    def load_histories(path, skip_assert=False):
        files = os.listdir(path)
        history_file_paths = filter(lambda f: "history" in f, files)
        history_file_paths = filter(lambda f: f.endswith(".pkl"), history_file_paths)
        history_file_paths = map(lambda f: os.path.join(path, f), history_file_paths)

        history_files = map(lambda p: HistoryState.load(p, skip_assert=skip_assert), history_file_paths)
        history_files = list(history_files)

        if len(history_files) < 1:
            raise Exception("Could not Find History File! (File must Contain 'history' and End with '.pkl')")

        if len(history_files) == 1:
            state = history_files[0]
            params = state.params
            history_history = state.history_history
            epochs = state.epoch

            return params, epochs, history_history

        history_files = sorted(history_files, key=lambda o: o.epoch[-1])
        history_files = list(history_files)

        latest_history = history_files[-1]

        params = latest_history.params
        values_by_ep_metric = defaultdict(lambda: defaultdict(lambda: []))

        def flatten_history_epoch(history):
            d = {}
            for loss_name, loss_values in history.history_history.items():
                d[loss_name] = {}

                for idx, loss in enumerate(loss_values):
                    ep = history.epoch[idx]
                    d[loss_name][str(ep)] = loss
            data = []
            for ep in history.epoch:
                ep_data = {}
                for loss in history.history_history.keys():
                    ep_data[loss] = d[loss][str(ep)]
                data.append((ep, ep_data))

            return data

        epoch_values = map(flatten_history_epoch, history_files)
        epoch_values = flatten(epoch_values)

        for epoch, values in epoch_values:
            for metric, metric_values in values.items():
                assert not isinstance(metric_values, list)
                values_by_ep_metric[epoch][metric].append(metric_values)

        values_by_ep_metric = dict(collections.OrderedDict(sorted(values_by_ep_metric.items())))
        values_by_ep_metric = {k: dict(v) for k, v in values_by_ep_metric.items()}

        epochs = []
        history = defaultdict(lambda: [])
        for ep, values in values_by_ep_metric.items():
            epochs.append(ep)
            for metric, metric_values in values.items():
                metric_values = distinct(metric_values)

#                assert len(metric_values) ==1, f"len(metric_values) = {len(metric_values)}, " \
#                         f"Path = {path}, metric_values = {metric_values}, Epoch = {ep}"
                if len(metric_values) == 1:
                    metric_value = metric_values[0]
                else:
                    print(f"Warning Averaging {len(metric_values)} Values to One Value")
                    metric_value = sum(metric_values) / len(metric_values)
                history[metric].append(metric_value)
        number_elements = distinct([len(x) for x in history.values()])

        if not skip_assert:
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
    p = 'D:\\masterarbeit_runs\\231_triplet_ctl_t\\history_files'
    print(HistoryState.load(p))




