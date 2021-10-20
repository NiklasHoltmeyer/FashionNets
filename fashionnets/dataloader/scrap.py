# Some Random Stuff to Clean up Notebooks
from fashionscrapper.default_logger.defaultLogger import defaultLogger

from fashionnets.networks.SiameseNetwork import SiameseNetwork

from fashionnets.models.embedding.simple_cnn import SimpleCNN

from fashionnets.models.embedding.resnet50 import ResNet50Builder

from fashionnets.dataloader.own import load_train_job
from fashionnets.dataloader.own import load_train_job
from fashionnets.models.embedding.simple_cnn import SimpleCNN
from fashionnets.models.SiameseModel import SiameseModel
from fashionnets.networks.SiameseNetwork import SiameseNetwork
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from fashionnets.models.embedding.resnet50 import ResNet50Builder
from fashionnets.callbacks.custom_save_weights import *
from fashionnets.callbacks.callbacks import callbacks

def _global_settings(notebook, epochs=50, nrows=None, verbose=False, buffer_size=32, batch_size=32):
  return {
      "batch_size": batch_size,
      "buffer_size": buffer_size,
      "notebook": notebook,
      "verbose": verbose,
      "nrows": nrows,
      "epochs": epochs,
  }


def load_job(back_bone_name, is_triplet, weights, input_shape, global_settings):
    alpha = 1.0
    beta = 0.5

    if "resnet50" == back_bone_name:
        run_name, back_bone, preprocess_input = job_resnet(input_shape, back_bone_name, is_triplet, weights)
    elif "simplecnn" == back_bone_name:
        run_name, back_bone, preprocess_input = job_simple_cnn(input_shape, back_bone_name, is_triplet)

    local_settings = {
        "alpha": alpha, "beta": beta,
        "triplets": is_triplet, "is_triplet": is_triplet,
        "input_shape": input_shape,
        "back_bone": back_bone,
        "preprocess_input": preprocess_input
    }
    _format = "triplet" if is_triplet else "quadruplet"
    d = load_train_job(run_name, format=_format, preprocess_input=preprocess_input,
                       **global_settings, target_shape=input_shape)
    return {**global_settings, **d, **local_settings}


def job_resnet(input_shape, back_bone, is_triplet, weights="mobile_net"):
    _format = "triplet" if is_triplet else "quadruplet"
    run_name = f"{back_bone}_{weights}_{_format}"

    back_bone, preprocess_input = ResNet50Builder.build(input_shape)
    if not weights:
        preprocess_input = None

    return run_name, back_bone, preprocess_input


def job_simple_cnn(input_shape, back_bone, is_triplet):
    weights = "none"
    _format = "triplet" if is_triplet else "quadruplet"
    run_name = f"{back_bone}_{weights}_{_format}"

    back_bone, preprocess_input = SimpleCNN.build(input_shape), None

    return run_name, back_bone, preprocess_input

def list_jobs():
    jobs = [
        {"back_bone_name": "resnet50", "weights": "mobile_net", "is_triplet": True},
        {"back_bone_name": "resnet50", "weights": None, "is_triplet": True},
        {"back_bone_name": "simplecnn", "weights": None, "is_triplet": True},
        {"back_bone_name": "resnet50", "weights": "mobile_net", "is_triplet": False},
        {"back_bone_name": "resnet50", "weights": None, "is_triplet": False},
        {"back_bone_name": "simplecnn", "weights": None, "is_triplet": False},
    ]

    return jobs, (144, 144)

def load_siamese_model(train_job, input_shape, keep_n=1, optimizer=None, verbose=False, result_uploader=None):
    logger = defaultLogger("Load_Siamese_Model")
    logger.disabled = not verbose

    if not optimizer:
        optimizer = Adam(1e-3)
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

    print("TODO"*10)
    print("Load Model")
    print("TODO" * 10)

    return siamese_model, _callbacks