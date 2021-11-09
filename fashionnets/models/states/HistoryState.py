import pickle

from tensorflow.keras import backend


class HistoryState:
    def __init__(self, history):
        self.params = backend.get_value(history.params)
        self.history_history = backend.get_value(history.history)
        self.epoch = backend.get_value(history.epoch)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            state = pickle.load(f)
            return state

    def apply(self, model):
        model.history.params = self.params
        model.history.history = self.history_history
        model.history.epoch = self.epoch
