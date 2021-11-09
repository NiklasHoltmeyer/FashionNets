import os
import pickle
import zipfile
from pathlib import Path

import tensorflow as tf

from fashionnets.callbacks.delete_checkpoints import DeleteOldModel
from fashionnets.train_jobs.loader.path_loader import _load_checkpoint_path


def load_latest_checkpoint(model, ignore_remote=False, **train_job):
    _checkpoint = None

    if not ignore_remote:
        _checkpoint, last_epoch = remote_checkpoint(train_job["environment"])

    if not _checkpoint:
        _checkpoint = latest_checkpoint(train_job["path"]["checkpoint"])
        last_epoch = retrieve_epoch_from_checkpoint(_checkpoint)

    if not _checkpoint:
        print("No Checkpoint found!")
        return False, 0

    checkpoint_file_path = Path(_checkpoint)

    if not checkpoint_file_path.exists():
        print("Checkpoint Path doesn not exist!")
        return False, 0

    opt_path = latest_optimizer(checkpoint_path=train_job["path"]["checkpoint"], epoch=last_epoch)

    model.load_embedding_weights(str(checkpoint_file_path.resolve()))
    model.make_train_function()

    with open(opt_path, 'rb') as f:
        weight_values = pickle.load(f)

    model.optimizer.set_weights(weight_values)

    return True, last_epoch + 1


def latest_checkpoint(checkpoint_path):
    valid_suffixes = [".ckpt", ".h5"]
    for suffix in valid_suffixes:
        latest_cp = sorted(filter(lambda d: d.endswith(suffix), os.listdir(checkpoint_path)))[-1:]

        if len(latest_cp) == 1:
            return os.path.join(checkpoint_path, latest_cp[0])

        return tf.train.latest_checkpoint(checkpoint_path)


def retrieve_epoch_from_checkpoint(latest_cp):
    if latest_cp is None:
        return None

    valid_suffixes = [".ckpt", ".h5"]
    for suffix in valid_suffixes:
        if latest_cp.endswith(suffix):
            fName = Path(latest_cp).name
            epoch_str = fName.split("-")[-1].split(".")[0]
            return int(epoch_str)
    return None


def latest_optimizer(checkpoint_path, epoch):
    epoch_str = f"{epoch:04d}"
    pickle_objects = filter(lambda p: p.endswith(".pkl"), os.listdir(checkpoint_path))
    optimizers = filter(lambda opt: epoch_str in opt, pickle_objects)
    optimizers = list(optimizers)

    assert len(optimizers) == 1, f"""{checkpoint_path} Contains {len(optimizers)} Optimizer!
    Looking for Optimizer with Epoch: {epoch} ({epoch_str}).
    *.pkl Objects: {list(filter(lambda p: p.endswith(".pkl"), os.listdir(checkpoint_path)))}
    """

    optimizer = optimizers[0]
    return str(Path(checkpoint_path, optimizer).resolve())


def remote_checkpoint(env):
    checkpoint_path = download_checkpoint(env)
    print("Downloaded Remote : ")

    if not checkpoint_path:
        return None, 0

    latest_cp = tf.train.latest_checkpoint(checkpoint_path)

    if not latest_cp:
        return None, 0

    last_epoch = retrieve_epoch_from_checkpoint(latest_cp)

    return latest_cp, last_epoch


def download_checkpoint(env):
    if not env.webdav:
        env.init()
    remote = env.webdav

    remote_base_path = remote.base_path.replace("//", "/")

    latest_zip = get_latest_zip_path(remote, remote_base_path)

    if not latest_zip:
        return

    zip_path = download_zip(env, latest_zip)
    return extract_zip(zip_path, env)


def extract_zip(zip_path, env):
    checkpoint_path = _load_checkpoint_path(env.train_job_name, notebook=env.notebook)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(checkpoint_path)

#    backup_files_local = os.listdir(checkpoint_path)
#    unnecessary_files = filter(lambda d: "backbone" in d, backup_files_local)
#    unnecessary_files = list(unnecessary_files)
#    unnecessary_files.append(zip_path)
    list(map(lambda d: DeleteOldModel.delete_path(str(Path(checkpoint_path, d).resolve())), unnecessary_files))

    return checkpoint_path


def download_zip(env, latest_zip):
    remote = env.webdav

    checkpoint_path = _load_checkpoint_path(env.train_job_name, notebook=env.notebook)

    zip_file_name = Path(latest_zip).name
    dst = Path(checkpoint_path, zip_file_name)

    if dst.exists():
        print("RETURN TODO")

    dst = str(dst.parent.resolve()) + ".zip"

    callback = lambda: print(f"Downloading Checkpoint: {zip_file_name} [DONE]")
    print(f"Downloading Checkpoint: {Path(latest_zip).name}")

    zip_path = remote.download(latest_zip, dst, callback, _async=False)

    return zip_path


def get_latest_zip_path(remote, remote_base_path):
    """

    :param remote: Remote-Client (e.G. WebDav)
    :param remote_base_path:
    :return: Path to Latest backup.zip
    """
    zips = remote.list(remote_base_path, _filter=lambda d: d.endswith(".zip"))
    zips = sorted(zips)

    if len(zips) < 1:
        return None

    latest_zip = sorted(zips)[-1]

    return "/".join([remote_base_path, latest_zip]).replace("//", "/")
