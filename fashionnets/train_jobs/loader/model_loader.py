from pathlib import Path

from fashiondatasets.utils.logger.defaultLogger import defaultLogger
from tensorflow import keras

from fashionnets.callbacks.callbacks import callbacks
from fashionnets.models.SiameseModel import SiameseModel
from fashionnets.networks.SiameseNetwork import SiameseNetwork
from fashionnets.util.csv import HistoryCSVHelper


def get_checkpoint(train_job, logger):
    cp_path, name = train_job["path"]["checkpoint"], train_job["run"]["name"]

    last_epoch = HistoryCSVHelper.last_epoch_from_train_job(train_job)
    init_epoch = last_epoch + 1

    if init_epoch > 0:
        checkpoint = str(Path(cp_path, name)) + f"_cp-{init_epoch:04d}.ckpt"

        logger.debug("Resume Training:")
        logger.debug(f" - Initial Epoch: {init_epoch}")
        logger.debug(f" - Checkpoint:    {checkpoint}")

        return init_epoch, checkpoint
    else:
        logger.debug(f"No Checkpoints found in: {cp_path}")
    return init_epoch, None


def load_siamese_model(train_job, input_shape, keep_n=1, optimizer=None, verbose=False, result_uploader=None):
    logger = defaultLogger("Load_Siamese_Model")
    logger.disabled = not verbose

    if not optimizer:
        optimizer = keras.optimizers.Adam(1e-3)
        logger.debug("Default Optimizer: Adam(LR 1e-3)")
    else:
        logger.debug("Using non Default Optimizer")

    parm_list = ["back_bone", "triplets", "input_shape", "alpha", "beta", "verbose", "channels"]
    kwargs = {k: train_job[k] for k in parm_list if train_job.get(k, None)}
    kwargs["triplets"] = kwargs.get("triplets", train_job["triplets"])
    logger.debug(f"Loading Siamese Network: {kwargs}")
    siamese_network = SiameseNetwork.build(**kwargs)

    siamese_model = SiameseModel(siamese_network)
    siamese_model.compile(optimizer=optimizer)
    siamese_model.fake_predict(input_shape, train_job["is_triplet"])

    cp_path, run_name = train_job["path"]["checkpoint"], train_job["run"]["name"]

    logger.debug(f"Checkpoint: {cp_path}")
    logger.debug(f"Run Name: {run_name}")

    _callbacks = callbacks(cp_path, run_name, monitor='val_loss', keep_n=keep_n,
                           verbose=verbose, result_uploader=result_uploader)

    logger.debug(f"Callbacks: {_callbacks}")

    init_epoch, _checkpoint = get_checkpoint(train_job, logger)
    if _checkpoint:
        logger.debug("Loading Weights!")
        siamese_model.load_weights(_checkpoint)

    return siamese_model, init_epoch, _callbacks
