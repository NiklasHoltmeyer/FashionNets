import os
import shutil
from pathlib import Path

from numba.core.typing.builtins import Zip
from tensorflow import keras

from fashionnets.callbacks.delete_checkpoints import DeleteOldModel
from fashionnets.util.io import read_file, write_file


class UploadResults(keras.callbacks.Callback):
    def __init__(self, checkpoint_path, result_uploader):
        super(UploadResults, self).__init__()
        self.checkpoint_path = checkpoint_path
        self.result_uploader = result_uploader

    def on_epoch_end(self, epoch, logs=None):
        zip_name = self.checkpoint_path + f"{epoch:04d}.zip"

        self.result_uploader.move(zip_name, _async=True)  # Upload Current Epoch Async

        for old_ep in range(max(epoch - 2, 0)):
            zip_name_old = self.checkpoint_path + f"{old_ep:04d}.zip"
            self.result_uploader.move(zip_name_old, _async=False)  # <- Retry uploading old Zips