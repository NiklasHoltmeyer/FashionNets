from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from fashionnets.callbacks.custom_save_weights import CustomSaveModel, CustomSaveWeights
from fashionnets.callbacks.delete_checkpoints import DeleteOldModel


def callbacks(checkpoint_path, name, monitor='val_loss', save_format=None, save_weights_only=False, keep_n=2):
    #save_callback = CustomSaveModel(checkpoint_path, name) if not save_weights_only else CustomSaveWeights(checkpoint_path, name)

    return [
        keras.callbacks.EarlyStopping(monitor='loss', patience=3),
        *model_checkpoint(checkpoint_path, name, monitor),
        DeleteOldModel(checkpoint_path=checkpoint_path, name=name, keep_n=keep_n, save_format=save_format),
    ]

def model_checkpoint(checkpoint_path, name, monitor='val_accuracy'):
    model_cp_path = Path(checkpoint_path, f"{name}_best_only.h5")
    model_cp_latest_path = Path(checkpoint_path, name+"_latest_ep{epoch}.h5")
    history_cp_path = Path(checkpoint_path, "history.csv")

    model_cp_path.parent.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.CSVLogger(history_cp_path, append=True, separator=";"),
        keras.callbacks.ModelCheckpoint(model_cp_path, save_best_only=True, monitor=monitor),
        keras.callbacks.ModelCheckpoint(model_cp_latest_path, monitor=monitor),
    ]

