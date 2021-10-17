from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from fashionnets.callbacks.custom_save_weights import CustomSaveModel, CustomSaveWeights
from fashionnets.callbacks.delete_checkpoints import DeleteOldModel
from fashionnets.callbacks.zip_results import ZipResults


def callbacks(checkpoint_path, name, monitor='val_loss', save_format=None, save_weights_only=False, keep_n=2,remove_after_zip=True):
    #save_callback = CustomSaveModel(checkpoint_path, name) if not save_weights_only else CustomSaveWeights(checkpoint_path, name)

    return [
        keras.callbacks.EarlyStopping(monitor='loss', patience=3),
        *model_checkpoint(checkpoint_path, name, monitor, save_weights_only=save_weights_only),
        DeleteOldModel(checkpoint_path=checkpoint_path, name=name, keep_n=keep_n,
                       save_format=save_format, save_weights_only=save_weights_only),
        ZipResults(checkpoint_path=checkpoint_path,remove_after_zip=remove_after_zip)
    ]

def model_checkpoint(checkpoint_path, name, monitor='val_accuracy', save_weights_only=False):
    model_cp_path = Path(checkpoint_path, f"{name}_best_") # .h5
    model_cp_latest_path = Path(checkpoint_path, name+"_") #.h5

    model_cp_path.parent.mkdir(parents=True, exist_ok=True)

    model_cp_path = str(model_cp_path.resolve()) + "cp-{epoch:04d}.ckpt"
    model_cp_latest_path = str(model_cp_latest_path.resolve()) + "cp-{epoch:04d}.ckpt"

    print("model_cp_path", model_cp_path)
    history_cp_path = Path(checkpoint_path, "history.csv")

    return [
        tf.keras.callbacks.CSVLogger(history_cp_path, append=True, separator=";"),
        keras.callbacks.ModelCheckpoint(model_cp_path, save_best_only=True, monitor=monitor,
                                        save_weights_only=save_weights_only),
        keras.callbacks.ModelCheckpoint(model_cp_latest_path, monitor=monitor,
                                        save_weights_only=save_weights_only),
    ]
