from matplotlib import pyplot as plt


def prepare_history(history):
    """

    :param history: History or HistoryState Object
    :return:
    """
    _history = getattr(history, "history", None)  # <- only works for History
    if not _history:
        _history = getattr(history, "history_history")  # <- only works for HistoryState

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


def plot_history(history, title=None, x_label=None, y_label=None, loc=None, colors=None,
                 label_mapping=None, build_epochs=None, bin_size=5):
    if colors is None:
        colors = list(reversed(
            ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
             '#17becf']))

    if build_epochs is None:
        build_epochs = []

    if label_mapping is None:
        label_mapping = {
            "loss": "Train",
            "val_loss": "Validation"
        }

    if x_label is None:
        x_label = "Epoch"
    if y_label is None:
        y_label = "Loss"
    if loc is None:
        loc = 'upper left'

    x_values, y_data = prepare_history(history)

    for k, v in y_data.items():
        color = colors.pop()

        label = label_mapping.get(k, k)

        plt.plot(x_values, v["values"], label=label, color=color)
        plt.plot(v["min_idx"], v["min_value"], 'v', color=color)  #

        for b_epoch in build_epochs:
            plt.plot(b_epoch, v["values"][b_epoch], 'go', color=color)  # linewidth=2, markersize=12

    if title:
        plt.title(title)

    x_ticks = sorted(list(range(0, max(x_values) + 1, bin_size)))

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.xticks(x_ticks)
    plt.legend(loc=loc)

    plt.xticks(x_ticks)
    plt.show()
