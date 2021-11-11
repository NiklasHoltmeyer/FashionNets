from matplotlib import pyplot as plt


def prepare_history(history):
    """

    :param history: History or HistoryState Object
    :return:
    """
    _history = getattr(history, "history", None)  # <- only works for History
    if not _history:
        _history = getattr(history, "history_history") # <- only works for HistoryState

    values = {}
    for k, v in _history.items():
        values[k] = {
            "values": v,
            "min_value": min(v),
            "min_idx": v.index(min(v)),
            "max_value": max(v),
            "max_idx": v.index(max(v))
        }

    x = history.epoch
    return x, values

def plot_history(history, title=None, x_label=None, y_label=None, loc=None, colors=None):
    if colors is None:
        colors = list(reversed(
            ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
             '#17becf']))

    if x_label is None:
        x_label = "Epoch"
    if y_label is None:
        y_label = "Loss"
    if loc is None:
        loc = 'upper left'

    #fig, ax = plt.subplots()

    x_values, y_data = prepare_history(history)

    for k, v in y_data.items():
        color = colors.pop()
        plt.plot(x_values, v["values"], label=k, color=color)
        plt.plot(v["min_idx"], v["min_value"], 'go', label="Min", color=color)

    labels = [e for e in x_values]

    if title:
        plt.title(title)

    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend(loc=loc)

    plt.xticks(x_values, labels)
    plt.show()