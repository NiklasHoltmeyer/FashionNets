import os
import shutil
from pathlib import Path
from collections import defaultdict
import collections
from tensorflow import keras


class DeleteOldModel(keras.callbacks.Callback):
    def __init__(self, checkpoint_path, name, keep_n, save_format=None, save_weights_only=False):
        super(DeleteOldModel, self).__init__()
        assert save_format in ["tf", "h5", None]

        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.mkdir(exist_ok=True, parents=True)

        self.checkpoint_path = str(checkpoint_path.resolve())

        self.name = name
        self.save_format = save_format
        self.ext = f".{save_format}" if save_format else ""
        self.keep_n = keep_n
        self.only_weights = save_weights_only

    def on_epoch_end(self, epoch, logs=None):
        try:
            file_name_checkpoints = self.list_checkpoints()
        except:
            return

        if self.keep_n >= len(file_name_checkpoints):
            return

        best_cps, train_cp = self.sorted_checkpoints(file_name_checkpoints)
        best_cps, train_cp = best_cps[:-self.keep_n], train_cp[:-self.keep_n]

        for files_to_del in [best_cps, train_cp]:
            for ep, paths in files_to_del:
                for path in paths:
                    path = str(Path(path).resolve())
                    if DeleteOldModel.delete_path(path):
                        print(f"Removing CP (EP={ep}): ./{path.name}")
                    else:
                        print(f"Removing CP (EP={ep}): ./{path.name} FAILED!!!")

    @staticmethod
    def delete_path(path):
        blacklist = [".csv", ".json"]
        if any(filter(lambda bl: path.endswith(bl), blacklist)):
            return True
        try:
            path.unlink()
            return True
        except:
            pass
        try:
            shutil.rmtree(path)
            return True
        except:
            return False

    @staticmethod
    def sorted_checkpoints(checkpoints):
        best_cps, train_cp = defaultdict(lambda: []), defaultdict(lambda: [])

        for ep, paths in list(checkpoints.items()):
            for path in paths:
                if "_best_" in path:
                    best_cps[ep].append(path)
                else:
                    train_cp[ep].append(path)

        best_cps, train_cp = best_cps.items(), train_cp.items()

        return list(best_cps), list(train_cp)

    def list_checkpoints(self):
        cp_extensions = ["ckpt", "index", "data", "tf", "h5"]
        unpack_path = lambda p: (os.path.join(self.checkpoint_path, p), *os.path.splitext(p))
        remove_ext_col = lambda d: (d[0], d[1])
        ep_from_name = lambda n: n.split(".ckpt")[0].split("-")[1]  # cp-0001.ckpt -> 0001

        filename_to_ep = lambda d: (d[0], ep_from_name(d[1].replace(self.name, "")))

        def is_cp(d):
            ext = d[2]
            for cp_ext in cp_extensions:
                if cp_ext in ext:
                    return True
            return False

        a_path_name_ext = map(unpack_path, os.listdir(self.checkpoint_path))
        a_path_name_ext = filter(is_cp, a_path_name_ext)
        a_path_name_ext = map(remove_ext_col, a_path_name_ext)
        a_path_name_ext = map(filename_to_ep, a_path_name_ext)

        paths_by_ep = defaultdict(lambda: [])

        for a_path, epoch in a_path_name_ext:
            paths_by_ep[int(epoch)].append(a_path)

        return collections.OrderedDict(sorted(paths_by_ep.items()))


if __name__ == "__main__":
    dom = DeleteOldModel(checkpoint_path=r"F:\workspace\FashNets\1337_resnet50_None_triplet",
                         keep_n=1, name="1337_resnet50_None_triplet")
    dom.on_epoch_end(None, None)
