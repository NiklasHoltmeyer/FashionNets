import logging
from pathlib import Path

from fashionnets.train_jobs.environment.Environment_Builder import EnvironmentBuilder
from fashionnets.train_jobs.job_list import job_config_by_notebook_name
from fashionnets.train_jobs.loader.backbone_loader import load_backbone_info_resnet, load_backbone_info_simple_cnn, \
    format_name
from fashionnets.train_jobs.loader.dataset_loader import load_dataset_loader
from fashionnets.train_jobs.loader.path_loader import _load_checkpoint_path
from fashionnets.util.io import json_dump, string_serializer
from fashiondatasets.utils.logger.defaultLogger import defaultLogger

logger = defaultLogger("deepfashion_model_builder", level=logging.INFO)

def prepare_environment(notebook_name, debugging=False, **settings):
    environment = EnvironmentBuilder.by_runtime_notebook_name(notebook_name)
    environment.load_kaggle()

    training_job_cfg = {**job_config_by_notebook_name(notebook_name, debugging), **settings}
    environment.dependencies["kaggle"] = training_job_cfg["dataset"]

    return environment, training_job_cfg


def load_job_settings(environment, training_job_cfg, kaggle_downloader, ignore_exception=False):
    if kaggle_downloader:
        environment.load_dependencies(kaggle_downloader=kaggle_downloader)

    job_settings = add_back_bone_to_train_job(environment=environment, **training_job_cfg)
    try:
        job = load_train_job(**job_settings)
    except Exception as e:
        job = {}
        if not ignore_exception:
            raise e
        else:
            logger.error("Exception:")
            logger.error(e)

    return {**job_settings, **job}


# noinspection PyUnboundLocalVariable
def add_back_bone_to_train_job(environment, **settings):
    back_bone_name = settings["back_bone"]["info"]["back_bone_name"]
    back_bone_weights = settings["back_bone"]["info"]["weights"]
    back_bone_is_triplet = settings["back_bone"]["info"]["is_triplet"]
    settings["format"] = format_name(back_bone_is_triplet)
    settings["is_triplet"] = back_bone_is_triplet

    assert settings["is_triplet"] in [True, False]
    assert type(settings["is_triplet"]) == bool
    assert (settings["format"]) in ["quadruplet", "triplet"]

    input_shape = settings["input_shape"]

    assert len(input_shape) == 2

    if "resnet50" == back_bone_name:
        run_name, embedding_model, preprocess_input_layer = load_backbone_info_resnet(input_shape=input_shape,
                                                                                      back_bone_name=back_bone_name,
                                                                                      is_triplet=back_bone_is_triplet,
                                                                                      weights=back_bone_weights)
        assert preprocess_input_layer
    elif "simplecnn" == back_bone_name:
        assert not back_bone_weights
        # noinspection PyArgumentList
        run_name, embedding_model, preprocess_input_layer = load_backbone_info_simple_cnn(input_shape=input_shape,
                                                                                          back_bone=back_bone_name,
                                                                                          is_triplet=back_bone_is_triplet)
    settings["back_bone"]["embedding_model_no_input_layer"] = embedding_model
    settings["back_bone"]["preprocess_input_layer"] = preprocess_input_layer
    logger.debug(settings["back_bone"]["preprocess_input_layer"])

    infos = [
        settings['run_idx'],
        settings.get("format", None),
        "ctl" if settings.get("is_ctl", False) else "apn",
        "t" if settings["is_triplet"] else "q",
    ]

    infos = [str(x) for x in infos]

    run_name = "_".join(infos)

    settings["run_name"] = run_name

    environment.train_job_name = run_name
    environment.init()

    settings["environment"] = environment
    settings["notebook"] = environment.notebook

    return settings


def load_train_job(run_name, **kwargs):
    path = {
        "checkpoint": _load_checkpoint_path(run_name, **kwargs),
    }

    run = {
        "name": run_name,
        "dataset": load_dataset_loader(**kwargs)
    }

    return {
        "path": path,
        "run": run
    }


def dump_settings(job_settings):
    cp_path = job_settings["path"]["checkpoint"]
    cp_path = Path(cp_path)
    cp_path.mkdir(parents=True, exist_ok=True)
    path = cp_path / "train_settings.json"

    settings = string_serializer(job_settings)

    if job_settings.get("verbose", False):
        logger.debug(f"Dumbing {path}")
    json_dump(path, settings)


def history_to_csv_string(history, _print=True, decimal_separator=None, **job_settings):
    name = job_settings["run_name"]
    lr = job_settings["learning_rate"]
    ds = job_settings["dataset"]["name"]
    is_trip = job_settings["is_triplet"]
    rows = []
    for metric, values in history.history.items():
        if decimal_separator:
            values = [str(x).replace(".", decimal_separator) for x in values]

        values = [metric, name, is_trip, lr, ds] + values
        values = [str(x) for x in values]

        csv = ";".join(values)
        rows.append(csv)
        if _print:
            logger.info(csv)

    return rows
