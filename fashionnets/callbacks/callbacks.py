from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from fashionnets.callbacks.save.csv import CSVLogger
from fashionnets.callbacks.save.embedding_model import SaveEmbeddingModel
from fashionnets.callbacks.remote.upload_results import UploadResults
from fashionnets.callbacks.remote.zip_results import ZipResults
from fashionnets.callbacks.save.history_state import SaveHistoryState
from fashionnets.callbacks.save.optimizer_state import SaveOptimizerState

csv_sep = ";"


def callbacks(checkpoint_path, name, save_format=None, save_weights_only=False, keep_n=2,
              remove_after_zip=True, verbose=False, result_uploader=None):
    history_cp_path = Path(checkpoint_path, "history.csv")

    if verbose:
        print(f"save_format={save_format}, save_weights_only={save_weights_only}, ")
        print(f"keep_n={keep_n},remove_after_zip={remove_after_zip}, verbose={verbose}")

    return [  # history_path, monitor="loss", patience=3, sep=","
        tf.keras.callbacks.LambdaCallback(on_train_begin=lambda logs: history_cp_path.parent.mkdir(exist_ok=True)),
        keras.callbacks.EarlyStopping(monitor='loss', patience=3),
        tf.keras.callbacks.CSVLogger(history_cp_path, append=True, separator=csv_sep),
        #        EarlyStoppingBasedOnHistory(history_path=history_cp_path, monitor='loss', patience=3, sep=csv_sep),
        #        model_checkpoint(checkpoint_path, name, monitor),
        SaveEmbeddingModel(model_cp_path=checkpoint_path),
        SaveOptimizerState(checkpoint_path=checkpoint_path, name=name),
        SaveHistoryState(checkpoint_path=checkpoint_path, name=name),
        CSVLogger(checkpoint_path=checkpoint_path, sep=csv_sep, decimal_symbol="."),
        ZipResults(checkpoint_path=checkpoint_path, remove_after_zip=remove_after_zip),
        UploadResults(checkpoint_path=checkpoint_path, result_uploader=result_uploader)
    ]


def model_checkpoint(checkpoint_path, name, monitor='val_accuracy'):
    model_cp_latest_path = Path(checkpoint_path, name + "_")  # .h5
    model_cp_latest_path.parent.mkdir(parents=True, exist_ok=True)
    model_cp_latest_path = str(model_cp_latest_path.resolve()) + "cp-{epoch:04d}.ckpt"

    return keras.callbacks.ModelCheckpoint(model_cp_latest_path, monitor=monitor)
