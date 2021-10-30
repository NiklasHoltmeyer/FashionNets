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
            fname_checkpoints = self.list_checkpoints()
        except:
            return

        if self.keep_n >= len(fname_checkpoints):
            return

        best_cps, train_cp = self.sorted_checkpoints(fname_checkpoints)
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

    def sorted_checkpoints(self, checkpoints):
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
        cp_exts = ["ckpt", "index", "data", "tf", "h5"]
        unpack_path = lambda p: (os.path.join(self.checkpoint_path, p), *os.path.splitext(p))
        remove_ext_col = lambda d: (d[0], d[1])
        ep_from_name = lambda n: n.split(".ckpt")[0].split("-")[1]  # cp-0001.ckpt -> 0001

        fname_to_ep = lambda d: (d[0], ep_from_name(d[1].replace(self.name, "")))

        def is_cp(d):
            ext = d[2]
            for cp_ext in cp_exts:
                if cp_ext in ext:
                    return True
            return False

        apath_name_ext = map(unpack_path, os.listdir(self.checkpoint_path))
        apath_name_ext = filter(is_cp, apath_name_ext)
        apath_name_ext = map(remove_ext_col, apath_name_ext)
        apath_name_ext = map(fname_to_ep, apath_name_ext)

        paths_by_ep = defaultdict(lambda: [])

        for apath, epoch in apath_name_ext:
            paths_by_ep[int(epoch)].append(apath)

        return collections.OrderedDict(sorted(paths_by_ep.items()))


if __name__ == "__main__":
    dom = DeleteOldModel(checkpoint_path=r"F:\workspace\FashNets\1337_resnet50_None_triplet",
                         keep_n=1, name="1337_resnet50_None_triplet")
    dom.on_epoch_end(None, None)

#    def on_epoch_end(self, epoch, logs=None):
#        fname_checkpoints = self.list_checkpoints()
#
#        if self.keep_n >= len(fname_checkpoints):
#            return

#        if not self.only_weights:
#            clean_name = lambda n: int(n.replace(self.name, "").replace(self.ext, "").split("_ep")[-1])
#        else:
#            clean_name = lambda x: x.split(".")[0].split("_ep")[-1]

#        get_ep = lambda d: (clean_name(d[0]), d[1])
#        fname_checkpoints = list(map(get_ep, fname_checkpoints))
#        sorted_cps = sorted(fname_checkpoints, key=lambda tup: tup[0])

#        delete_cps = list(sorted_cps)

#        if not self.only_weights:
#            for ep, path in delete_cps[:-self.keep_n]:
#                print(f"Removing CP (EP={ep}): ./{path.name}")
#                path.unlink()
#        else:
#            paths_by_ep = defaultdict(lambda: [])
#            for ep, path in delete_cps:
#                paths_by_ep[ep].append(path)

#            paths_by_ep = collections.OrderedDict(sorted(paths_by_ep.items()))
#            to_delete = list(paths_by_ep.items())[:-self.keep_n]
#            for ep, paths in to_delete:
#                for p in paths:
#                    print(f"Removing CP (EP={ep}): ./{p.name}")
#                    p.unlink()

#    def list_checkpoints(self, keep_best=True):
#        if not self.only_weights:
#            is_checkpoint = lambda x: x[-3:] == self.ext
#        else:
#            is_checkpoint = lambda x: "_ep" in x and (x.endswith(".index") or ".data-" in x)
#        is_not_best = lambda x: not "_best_" in x

#        checkpoints = filter(is_checkpoint, os.listdir(self.checkpoint_path))
#        if keep_best:
#            checkpoints = filter(is_not_best, checkpoints)
#        f_names = list(checkpoints)
#        fullpaths = list(map(lambda p: Path(self.checkpoint_path + f"/{p}"), f_names))
#        return list(zip(f_names, fullpaths))
