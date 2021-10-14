import os
from pathlib import Path

from tensorflow import keras


class DeleteOldModel(keras.callbacks.Callback):
    def __init__(self, checkpoint_path, name, keep_n, save_format=None):
        super(DeleteOldModel, self).__init__()
        assert save_format in ["tf", "h5", None]

        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.mkdir(exist_ok=True, parents=True)

        self.checkpoint_path = str(checkpoint_path.resolve())

        self.name = name
        self.save_format = save_format
        self.ext = f".{save_format}" if save_format else ""
        self.keep_n = keep_n

    def on_epoch_end(self, epoch, logs=None):
        fname_checkpoints = self.list_checkpoints()

        if self.keep_n >= len(fname_checkpoints):
            return

        clean_name = lambda n: int(n.replace(self.name, "").replace(self.ext, "").split("_ep")[-1])
        get_ep = lambda d: (clean_name(d[0]), d[1])
        fname_checkpoints = list(map(get_ep, fname_checkpoints))
        sorted_cps = sorted(fname_checkpoints, key=lambda tup: tup[0])

        delete_cps = (list(sorted_cps)[:-self.keep_n])

        for ep, path in delete_cps:
            print(f"Removing CP (EP={ep}): ./{path.name}")
            path.unlink()

    def list_checkpoints(self, keep_best=True):
        is_checkpoint = lambda x: x[-3:] == self.ext
        is_not_best = lambda x: not "best_only" in x

        checkpoints = filter(is_checkpoint, os.listdir(self.checkpoint_path))
        if keep_best:
            checkpoints = filter(is_not_best, checkpoints)
        f_names = list(checkpoints)
        fullpaths = list(map(lambda p: Path(self.checkpoint_path + f"/{p}"), f_names))
        return list(zip(f_names, fullpaths))
