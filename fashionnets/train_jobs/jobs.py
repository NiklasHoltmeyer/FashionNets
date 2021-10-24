from pathlib import Path

from fashionnets.train_jobs.loader.backbone_loader import *
from fashionnets.train_jobs.loader.dataset_loader import loader_info, load_dataset_loader
from fashionnets.train_jobs.loader.job_loader import _load_checkpoint_path
from fashionnets.train_jobs.notebook_environments.env_loader import env_by_name
from fashionnets.util.io import json_dump


def job_list():
    global_settings = {
        "input_shape": (224, 224),#(144, 144),
        "alpha": 1.0,
        "beta": 0.5,
        "epochs": 20,
        "verbose": False,
        "nrows": None,
        "buffer_size": 32,
        "batch_size": 32
    }

    debugging_settings = {
        "nrows": 20,
        "verbose": True,
        "batch_size": 1,
        "epochs": 5
    }

    back_bone_variants = [
        {"back_bone_name": "resnet50", "weights": "imagenet", "is_triplet": True, **global_settings},
        {"back_bone_name": "resnet50", "weights": None, "is_triplet": True, **global_settings},
        {"back_bone_name": "simplecnn", "weights": None, "is_triplet": True, **global_settings},
#        {"back_bone_name": "resnet50", "weights": "imagenet", "is_triplet": False, **global_settings},
#        {"back_bone_name": "resnet50", "weights": None, "is_triplet": False, **global_settings},
#        {"back_bone_name": "simplecnn", "weights": None, "is_triplet": False},
    ]

    #    datasets = {
    #        "masterokay/own-sl-256": None,
    #        "deep_fashion_256": [
    #            "df_quad_1",
    #            "df_quad_2",
    #            "df_quad_3",
    #        ]
    #    }

    # ds_name = "own_256" <- "masterokay/own-sl-256"
#    ds_info_variants = [
#        loader_info("deep_fashion_256", "df_quad_3"), #3, A = User, P / N1 = Shop, N2 = User
#        loader_info("deep_fashion_256", "df_quad_2"), #2, A = User, P / N1 / N2 = Shop
#        loader_info("deep_fashion_256", "df_quad_1"), #1, A = User, P = Shop, N1 / N2 -> 50 / 50
#    ]
#    ds_info = loader_info("deep_fashion_256", "df_quad_3")

#-> back_bone_variants[-2 / 0] funkt nicht

    return {
#        "g_i": {**back_bone_variants[1], "dataset": loader_info("deep_fashion_256", "df_quad_3"), "run_idx": 6},
#        "k_ok": {**back_bone_variants[1], "dataset": loader_info("deep_fashion_256", "df_quad_2"), "run_idx": 7},
#        "g_v": {**back_bone_variants[1], "dataset": loader_info("deep_fashion_256", "df_quad_1"), "run_idx": 8},

#        "g_p": {**back_bone_variants[3], "dataset": loader_info("deep_fashion_256", "df_quad_3"), "run_idx": 9},
#        "g_b":  {**back_bone_variants[3], "dataset": loader_info("deep_fashion_256", "df_quad_2"), "run_idx": 10},
#        "g_ok": {**back_bone_variants[3], "dataset": loader_info("deep_fashion_256", "df_quad_1"), "run_idx": 11},

#        "l_h": {**back_bone_variants[0], "dataset": loader_info("deep_fashion_256", "df_quad_1"), "run_idx": 1337},

#        "g_i1": {**back_bone_variants[0], "dataset": loader_info("deep_fashion_256", "df_quad_3"), "run_idx": 1337},
#        "g_i2": {**back_bone_variants[1], "dataset": loader_info("deep_fashion_256", "df_quad_3"), "run_idx": 1337},
#        "g_i3": {**back_bone_variants[2], "dataset": loader_info("deep_fashion_256", "df_quad_3"), "run_idx": 1337},
#        "g_i4": {**back_bone_variants[3], "dataset": loader_info("deep_fashion_256", "df_quad_3"), "run_idx": 1337},
#        "g_i5": {**back_bone_variants[4], "dataset": loader_info("deep_fashion_256", "df_quad_3"), "run_idx": 1337},
#        "g_i6": {**back_bone_variants[5], "dataset": loader_info("deep_fashion_256", "df_quad_3"), "run_idx": 1337},

        "l_i1": {**back_bone_variants[0], "dataset": loader_info("own", "df_quad_3"), "run_idx": 13370,
                 **debugging_settings}, #    return
        "l_i2": {**back_bone_variants[1], "dataset": loader_info("own", "df_quad_3"), "run_idx": 13371,
                 **debugging_settings},
        "l_i3": {**back_bone_variants[2], "dataset": loader_info("own", "df_quad_3"), "run_idx": 13372,
                 **debugging_settings},
    }


def load_job_info_from_notebook_name(notebook_name):
    env = env_by_name(notebook_name)

    # global_settings = global_settings(env.notebook)
    job_info = job_list().get(notebook_name, None)
    assert job_info, "Unknown Notebook Name"
    env.dependencies["kaggle"] = job_info["dataset"]

    assert job_info, "Job = None"

    job_settings = load_backbone_info(**job_info, environment=env)

    return job_settings


def load_train_job(name, **kwargs):
    path = {
        "checkpoint": _load_checkpoint_path(name, **kwargs),
    }

    run = {
        "name": name,
        "dataset": load_dataset_loader(**kwargs)
    }

    return {
        "path": path,
        "run": run
    }


def load_backbone_info(back_bone_name, is_triplet, weights, input_shape, alpha, beta, environment=None,
                       **settings):
    if "resnet50" == back_bone_name:
        run_name, back_bone, preprocess_input = load_backbone_info_resnet(input_shape, back_bone_name, is_triplet, weights)
    elif "simplecnn" == back_bone_name:
        run_name, back_bone, preprocess_input = load_backbone_info_simple_cnn(input_shape, back_bone_name, is_triplet)

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



