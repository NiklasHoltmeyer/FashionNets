from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from fashionnets.callbacks.delete_checkpoints import DeleteOldModel
from fashionnets.callbacks.early_stopping_based_on_history_csv import EarlyStoppingBasedOnHistory
from fashionnets.callbacks.zip_results import ZipResults

csv_sep = ";"


def callbacks(checkpoint_path, name, monitor='val_loss', save_format=None, save_weights_only=False, keep_n=2,
              remove_after_zip=True, verbose=False, result_uploader=None):
    # save_callback = CustomSaveModel(checkpoint_path, name) if not save_weights_only else CustomSaveWeights(checkpoint_path, name)
    history_cp_path = Path(checkpoint_path, "history.csv")

    if verbose:
        print(f"save_format={save_format}, save_weights_only={save_weights_only}, ")
        print(f"keep_n={keep_n},remove_after_zip={remove_after_zip}, verbose={verbose}")

    return [  # history_path, monitor="loss", patience=3, sep=","
        keras.callbacks.EarlyStopping(monitor='loss', patience=3),
        tf.keras.callbacks.CSVLogger(history_cp_path, append=True, separator=csv_sep),
        EarlyStoppingBasedOnHistory(history_path=history_cp_path, monitor='loss', patience=3, sep=csv_sep),
        *model_checkpoint(checkpoint_path, name, monitor, save_weights_only=save_weights_only, verbose=verbose),
        DeleteOldModel(checkpoint_path=checkpoint_path, name=name, keep_n=keep_n,
                       save_format=save_format, save_weights_only=save_weights_only),
        ZipResults(checkpoint_path=checkpoint_path, remove_after_zip=remove_after_zip, result_uploader=result_uploader),
    ]


def model_checkpoint(checkpoint_path, name, monitor='val_accuracy', save_weights_only=False, verbose=False):
    model_cp_path = Path(checkpoint_path, f"{name}_best_")  # .h5
    model_cp_latest_path = Path(checkpoint_path, name + "_")  # .h5

    model_cp_path.parent.mkdir(parents=True, exist_ok=True)

    model_cp_path = str(model_cp_path.resolve()) + "cp-{epoch:04d}.ckpt"
    model_cp_latest_path = str(model_cp_latest_path.resolve()) + "cp-{epoch:04d}.ckpt"

    if verbose:
        print("model_cp_path", model_cp_path)

    return [
        keras.callbacks.ModelCheckpoint(model_cp_path, save_best_only=True, monitor=monitor,
                                        save_weights_only=save_weights_only),
        keras.callbacks.ModelCheckpoint(model_cp_latest_path, monitor=monitor,
                                        save_weights_only=save_weights_only),
    ]
