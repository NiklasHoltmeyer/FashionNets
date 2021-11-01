import os
from pathlib import Path
import zipfile

from fashionnets.callbacks.delete_checkpoints import DeleteOldModel
from fashionnets.train_jobs.loader.path_loader import _load_checkpoint_path
import tensorflow as tf

def remote_checkpoint(env):
    checkpoint_path = download_checkpoint(env)
    latest_cp = tf.train.latest_checkpoint(checkpoint_path)

    if not latest_cp:
        return None, 0

    init_epoch = int(latest_cp.split("_cp-")[-1].replace(".ckpt", ""))

    return latest_cp, init_epoch

def download_checkpoint(env):
    if not env.webdav:
        env.init()
    remote = env.webdav

    remote_base_path = remote.base_path.replace("//", "/")

    return r"F:\workspace\FashNets\11_resnet50_None_quadruplet"

#    latest_zip = get_latest_zip_path(remote, remote_base_path)

#    if not latest_zip:
#        return

#    zip_path = download_zip(env, latest_zip)
#    return extract_zip(zip_path, env)


def extract_zip(zip_path, env):
    checkpoint_path = _load_checkpoint_path(env.train_job_name, notebook=env.notebook)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(checkpoint_path)

    backup_files_local = os.listdir(checkpoint_path)
    unnecessary_files = filter(lambda d: "backbone" in d, backup_files_local)
    unnecessary_files = list(unnecessary_files)
    unnecessary_files.append(zip_path)
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
