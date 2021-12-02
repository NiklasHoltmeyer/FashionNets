from matplotlib import pyplot as plt


def prepare_history(history, **kwargs):
    """

    :param history: History or HistoryState Object
    :return:
    """
    _history = getattr(history, "history", None)  # <- only works for History
    if not _history:
        _history = getattr(history, "history_history")  # <- only works for HistoryState

    xmax = kwargs.get("xmax", None)

    values = {}
    for k, v in _history.items():
        if xmax:
            v = v[:xmax]

        values[k] = {
            "values": v,
            "min_value": min(v),
            "min_idx": v.index(min(v)),
            "max_value": max(v),
            "max_idx": v.index(max(v))
        }

    x = history.epoch

    if xmax:
        x = x[:xmax]

    return x, values


def plot_history(histories, **kwargs):
    colors = kwargs.get("colors", list(reversed(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                                                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22','#17becf'])))
    build_epochs = kwargs.get("build_epochs", [])
    build_epochs_symbol = kwargs.get("build_epochs_symbol", 'go')
    #
    label_mapping = kwargs.get("label_mapping",  {"loss": "Train","val_loss": "Validation"})

    x_label = kwargs.get("x_label", "Epoch")
    y_label = kwargs.get("y_label", "Loss")
    loc = kwargs.get("loc", 'upper left')
    bin_size = kwargs.get("bin_size", 5)
    save_path = kwargs.get("save_path", None)
    title = kwargs.get("title", None)

    linewidth = kwargs.get("linewidth", 2)
    markersize = kwargs.get("markersize", 12)

    if type(histories) != list:
        histories = [histories]

    x_labels = kwargs.get("x_labels", [None] * len(histories))

    max_x = 0

    for history, lbl in zip(histories, x_labels):
        x_values, y_data = prepare_history(history, **kwargs)

        max_x = max(max_x, max(x_values))

        for k, v in y_data.items():
            color = colors.pop()

            if lbl:
                label = lbl
            else:
                label = label_mapping.get(k, k)

            plt.plot(x_values, v["values"], label=label, color=color)
            plt.plot(v["min_idx"], v["min_value"], 'v', color=color, linewidth=linewidth, markersize=12)  #

            for b_epoch in build_epochs:
                plt.plot(b_epoch, v["values"][b_epoch], build_epochs_symbol,
                         color=color, linewidth=linewidth, markersize=markersize)

    if title:
        plt.title(title)

    x_ticks = sorted(list(range(0, (max_x + 2), bin_size)))

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.xticks(x_ticks)
    plt.legend(loc=loc)

    plt.ylim(ymin=kwargs.get("ymin", None), ymax=kwargs.get("ymax", None))
    plt.xlim(xmin=kwargs.get("xmin", None), xmax=kwargs.get("xmax", None) + 2)

    plt.xticks(x_ticks)

    if save_path:
        plt.savefig(save_path)

    plt.show()
