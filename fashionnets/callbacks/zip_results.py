import os
import shutil
from pathlib import Path

from numba.core.typing.builtins import Zip
from tensorflow import keras

from fashionnets.callbacks.delete_checkpoints import DeleteOldModel
from fashionnets.util.io import read_file, write_file


class ZipResults(keras.callbacks.Callback):
    def __init__(self, checkpoint_path, remove_after_zip, result_uploader=None):
        super(ZipResults, self).__init__()
        self.checkpoint_path = checkpoint_path
        self.remove_after_zip = remove_after_zip
        self.result_uploader = result_uploader

    def on_epoch_end(self, epoch, logs=None):
        self.zip_results(self.checkpoint_path, True, epoch=epoch)

    def zip_results(self, folder_path, overwrite=False, epoch=None):
        if not overwrite and Path(folder_path + ".zip").exists():
            return
        print(f"Zipping: {folder_path}")
        try:
            zip_name = folder_path + f"{epoch:04d}"
            shutil.make_archive(zip_name, 'zip', folder_path)

            if self.remove_after_zip:
                self.delete_already_zipped_results(folder_path)
            print(self.result_uploader)
            print(self.result_uploader is not None)
            if self.result_uploader:
                self.result_uploader.move(zip_name+".zip")

        except Exception as e:
            print("zip_results Expcetion")
            print(e)

    def delete_already_zipped_results(self, folder_path):
        history_path = Path(folder_path, "history.csv")
        history = read_file(history_path) if history_path.exists() else None
        if not DeleteOldModel.delete_path(folder_path):
            print("Couldnt Remove:", folder_path)
        if history:
            history_path.parent.mkdir(parents=True, exist_ok=True)
            write_file(history_path, history)

    @staticmethod
    def list_subfolders(path):
        return [f.path for f in os.scandir(path) if f.is_dir()]

    @staticmethod
    def zip_subfolders(path):
        return list(map(ZipResults.zip_results, ZipResults.list_subfolders(path)))

if __name__ == "__main__":
    zip_r = ZipResults(checkpoint_path=r"F:\workspace\FashNets\1337_resnet50_None_triplet",
                         remove_after_zip=True)
    zip_r.on_epoch_end(None, None)
