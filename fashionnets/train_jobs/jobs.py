from pathlib import Path

from fashiondatasets.utils.logger.defaultLogger import defaultLogger
from tensorflow import keras

from fashionnets.callbacks.callbacks import callbacks
from fashionnets.models.SiameseModel import SiameseModel
from fashionnets.networks.SiameseNetwork import SiameseNetwork
from fashionnets.train_jobs.backbone_loader import *
from fashionnets.train_jobs.job_loader import _load_checkpoint_path, load_dataset
from fashionnets.train_jobs.notebook_environments import environment
from fashionnets.util.csv import HistoryCSVHelper
from fashionnets.util.io import json_dump


def job_list():
    global_settings = {
        "input_shape": (144, 144),
        "alpha": 1.0,
        "beta": 0.5,
        "epochs": 25,
        "verbose": True,
        "nrows": None,
        "buffer_size": 32,
        "batch_size": 32
    }

    back_bone_variants = [
        {"back_bone_name": "resnet50", "weights": "mobile_net", "is_triplet": True, **global_settings},
        {"back_bone_name": "resnet50", "weights": None, "is_triplet": True, **global_settings},
        #        {"back_bone_name": "simplecnn", "weights": None, "is_triplet": True, **global_settings},
        {"back_bone_name": "resnet50", "weights": "mobile_net", "is_triplet": False, **global_settings},
        {"back_bone_name": "resnet50", "weights": None, "is_triplet": False, **global_settings},
        #        {"back_bone_name": "simplecnn", "weights": None, "is_triplet": False},
    ]

    #    datasets = {
    #        "masterokay/own-sl-256": None,
    #        "deep_fashion_256": [
    #            "df_quad_1", #1, A = User, P = Shop, N1 / N2 -> 50 / 50
    #            "df_quad_2", #2, A = User, P / N1 / N2 = Shop
    #            "df_quad_3", #3, A = User, P / N1 = Shop, N2 = User
    #        ]
    #    }

    # ds_name = "own_256" <- "masterokay/own-sl-256"

    ds = {
        "name": "deep_fashion_256",
        "variation": "df_quad_3",
        "cleanup_ops": [
            ("rename", "./df_quad_3", "./deep_fashion"),
            ("mv", "./train_256/images", "./deep_fashion/train"),
            ("mv", "./validation_256/images", "./deep_fashion/validation"),
            ("rm", "./validation_256", None),
            ("rm", "./train_256", None),
            ("mv", "./deep_fashion", "./deep_fashion_256")
        ]  # src, dst
    }

    return {
        "g_i": {**back_bone_variants[-1], "dataset": ds, "run_idx": 0},
        "k_ok": {**back_bone_variants[-2], "dataset": ds, "run_idx": 1},
        "g_v": {**back_bone_variants[-3], "dataset": ds, "run_idx": 2},
        "g_p": {**back_bone_variants[-4], "dataset": ds, "run_idx": 3},

        "g_b": None,
        "g_ok": None,

        "l_h": {**back_bone_variants[0], "dataset": ds, "run_idx": 3}  # <- just dbug
    }


def load_job_from_notebook_name(notebook_name):
    env = environment(notebook_name)

    # global_settings = global_settings(env.notebook)
    job_info = job_list().get(notebook_name, None)

    env.dependencies["kaggle"] = job_info["dataset"]

    assert job_info, "Job = None"

    job_settings = load_job(**job_info, environment=env)

    return job_settings


def load_train_job(name, **kwargs):
    path = {
        "checkpoint": _load_checkpoint_path(name, **kwargs),
    }

    run = {
        "name": name,
        "dataset": load_dataset(**kwargs)
    }

    return {
        "path": path,
        "run": run
    }


def load_job(back_bone_name, is_triplet, weights, input_shape, alpha, beta, environment=None,
             **settings):
    if "resnet50" == back_bone_name:
        run_name, back_bone, preprocess_input = job_resnet(input_shape, back_bone_name, is_triplet, weights)
    elif "simplecnn" == back_bone_name:
        run_name, back_bone, preprocess_input = job_simple_cnn(input_shape, back_bone_name, is_triplet)

    run_name = "" + f"{settings['run_idx']}_{run_name}"

    local_settings = {
        "alpha": alpha, "beta": beta,
        "triplets": is_triplet, "is_triplet": is_triplet,
        "input_shape": input_shape,
        "back_bone": back_bone,
        "preprocess_input": preprocess_input
    }
    _format = "triplet" if is_triplet else "quadruplet"
    assert environment, "ENV must be Set!"

    environment.train_job_name = run_name
    environment.init()

    settings["environment"] = environment
    settings["notebook"] = environment.notebook

    return {**local_settings, **settings}

def load_job_f_settings(**settings):
    d = load_train_job(**settings)
    dump_settings(settings)

    return {**settings, **d}


def dump_settings(job_settings):
    cp_path = job_settings["path"]["checkpoint"]
    cp_path = Path(cp_path)
    cp_path.mkdir(parents=True, exist_ok=True)
    path = cp_path / "settings.json"

    settings = dict(job_settings)

    settings["preprocess_input"] = str(settings["preprocess_input"])
    settings["back_bone"] = str(settings["back_bone"])

    settings["run"]["dataset"]["train"] = str(
        settings["run"]["dataset"]["train"])  # <- JSON tries to Serialize the Data
    settings["run"]["dataset"]["val"] = str(settings["run"]["dataset"]["val"])
    settings["environment"] = str(settings["environment"])

    if job_settings.get("verbose", False):
        print(f"Dumbing {path}")
    json_dump(path, settings)


def get_checkpoint(train_job, logger):
    cp_path, name = train_job["path"]["checkpoint"], train_job["run"]["name"]

    last_epoch = HistoryCSVHelper.last_epoch_from_train_job(train_job)
    init_epoch = last_epoch + 1

    if init_epoch > 0:
        checkpoint = str(Path(cp_path, name)) + f"_cp-{init_epoch:04d}.ckpt"

        logger.debug("Resume Training:")
        logger.debug(f" - Initial Epoch: " + init_epoch)
        logger.debug(f" - Checkpoint:    " + checkpoint)

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
