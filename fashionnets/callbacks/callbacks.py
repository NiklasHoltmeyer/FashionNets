from pathlib import Path

import tensorflow as tf
from tensorflow import keras


def callbacks(checkpoint_path, name, monitor='val_accuracy'):
    return [
        *model_checkpoint(checkpoint_path, name, monitor),
        keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    ]

def model_checkpoint(checkpoint_path, name, monitor='val_accuracy'):
    model_cp_path = Path(checkpoint_path, f"{name}_best_only.h5")
    model_cp_latest_path = Path(checkpoint_path, name+"_latest_ep{epoch}.h5")
    history_cp_path = Path(checkpoint_path, "history.csv")

    model_cp_path.parent.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.ModelCheckpoint(model_cp_path, save_best_only=True, monitor=monitor),
        keras.callbacks.ModelCheckpoint(model_cp_latest_path, monitor=monitor),
        tf.keras.callbacks.CSVLogger(history_cp_path, append=True, separator=";"),
    ]

