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
    path = r"D:\Download\311_resnet50_imagenet_quadruplet_history-0004.pkl"
    print(HistoryState.load(path))