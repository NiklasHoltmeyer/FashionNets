import os
import shutil
from pathlib import Path

from numba.core.typing.builtins import Zip
from tensorflow import keras

from fashionnets.callbacks.delete_checkpoints import DeleteOldModel


class ZipResults(keras.callbacks.Callback):
    def __init__(self, checkpoint_path, remove_after_zip, force_delete_zip=True):
        super(ZipResults, self).__init__()
        self.checkpoint_path = checkpoint_path
        self.remove_after_zip = remove_after_zip
        self.force_delete_zip=force_delete_zip

    def on_epoch_end(self, epoch, logs=None):
        self.zip_results(self.checkpoint_path, True)

    def zip_results(self, folder_path, overwrite=False):
        if self.force_delete_zip:
            print("Force Folder", Path(folder_path + ".zip"))
            DeleteOldModel.delete_path(Path(folder_path + ".zip"))

        if not overwrite and Path(folder_path + ".zip").exists():
            return
        print(f"Zipping: {folder_path}")
        try:
            shutil.make_archive(folder_path, 'zip', folder_path)
            if self.remove_after_zip:
                if not DeleteOldModel.delete_path(folder_path):
                    print("Couldnt Remove:", folder_path)
        except Exception as e:
            print("zip_results Expcetion")
            print(e)

    @staticmethod
    def list_subfolders(path):
        return [f.path for f in os.scandir(path) if f.is_dir()]

    @staticmethod
    def zip_subfolders(path):
        return list(map(ZipResults.zip_results, ZipResults.list_subfolders(path)))
