import os
import shutil
from pathlib import Path

from numba.core.typing.builtins import Zip
from tensorflow import keras


class ZipResults(keras.callbacks.Callback):
    def __init__(self, checkpoint_path, remove_after_zip):
        super(ZipResults, self).__init__()
        self.checkpoint_path  = checkpoint_path
        self.remove_after_zip = remove_after_zip

    def on_epoch_end(self, epoch, logs=None):
        ZipResults.zip_results(self.checkpoint_path)
        pass

    @staticmethod
    def zip_results(folder_path, overwrite=False):
        if not overwrite and Path(folder_path + ".zip").exists():
            return
        print(f"Zipping: {folder_path}")
        shutil.make_archive(folder_path, 'zip', folder_path)

    @staticmethod
    def list_subfolders(path):
        return [f.path for f in os.scandir(path) if f.is_dir()]

    @staticmethod
    def zip_subfolders(path):
        return list(map(ZipResults.zip_results, ZipResults.list_subfolders(path)))
