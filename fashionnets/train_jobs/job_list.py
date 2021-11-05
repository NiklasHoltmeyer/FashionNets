from fashionscrapper.utils.list import distinct
from fashionnets.models.embedding.resnet50 import ResNet50Builder
from fashionnets.train_jobs.loader.dataset_loader import loader_info
from fashionnets.train_jobs.settings.default_settings import base_settings, back_bone_settings


def job_config_by_notebook_name(notebook_name, debugging):
    x = notebook_name.split("_")[1:]
    job_id = "_".join(x)  # g_b1_12 => b1_12
    return job_list(debugging)[job_id]

def job_list(debugging):
    base_cfg1e4 = base_settings(debugging, 1e-4)
    base_cfg5e3 = base_settings(debugging, 5e-3)
    base_cfg1e3 = base_settings(debugging, 1e-3)

    # back_bone_cfg =
    triplet_w_weights = back_bone_settings("resnet50", weights="imagenet", is_triplet=True)
#    triplet_no_weights = back_bone_settings("resnet50", weights=None, is_triplet=True)

    quad_w_weights = back_bone_settings("resnet50", weights="imagenet", is_triplet=False)
#    quad_no_weights = back_bone_settings("resnet50", weights=None, is_triplet=False)

    ds = loader_info("deep_fashion_1")

    back_bone_by_notebook = {
#        "b1": {**quad_no_weights},
        "q": {**quad_w_weights},

        "t": {**triplet_w_weights},
#        "b4": {**triplet_no_weights},
    }

    freeze_layers = {
        "non_conv5_block1_out": ResNet50Builder.freeze_non_conv5_block1_out,
        "first_100": lambda model: ResNet50Builder.freeze_first_n_layers(model, 100),
        "none": None
    }

    train_jobs = {
        #first 100
        "q_11_f100": {"run_idx": 111, **base_cfg1e4, "dataset": ds, "freeze_layers": freeze_layers["first_100"]},
        "t_12_f100": {"run_idx": 112, **base_cfg1e4, "dataset": ds, "freeze_layers": freeze_layers["first_100"]},

        "q_21_f100": {"run_idx": 121, **base_cfg5e3, "dataset": ds, "freeze_layers": freeze_layers["first_100"]},
        "t_22_f100": {"run_idx": 122, **base_cfg5e3, "dataset": ds, "freeze_layers": freeze_layers["first_100"]},

        "q_31_f100": {"run_idx": 131, **base_cfg1e3, "dataset": ds, "freeze_layers": freeze_layers["first_100"]},
        "t_32_f100": {"run_idx": 132, **base_cfg1e3, "dataset": ds, "freeze_layers": freeze_layers["first_100"]},

        #non conv5
        "q_11_conv5": {"run_idx": 211, **base_cfg1e4, "dataset": ds, "freeze_layers": freeze_layers["non_conv5_block1_out"]},
        "t_12_conv5": {"run_idx": 212, **base_cfg1e4, "dataset": ds, "freeze_layers": freeze_layers["non_conv5_block1_out"]},

        "q_21_conv5": {"run_idx": 221, **base_cfg5e3, "dataset": ds, "freeze_layers": freeze_layers["non_conv5_block1_out"]},
        "t_22_conv5": {"run_idx": 222, **base_cfg5e3, "dataset": ds, "freeze_layers": freeze_layers["non_conv5_block1_out"]},

        "q_31_conv5": {"run_idx": 231, **base_cfg1e3, "dataset": ds, "freeze_layers": freeze_layers["non_conv5_block1_out"]},
        "t_32_conv5": {"run_idx": 232, **base_cfg1e3, "dataset": ds, "freeze_layers": freeze_layers["non_conv5_block1_out"]},

        # None
        "q_11_none": {"run_idx": 311, **base_cfg1e4, "dataset": ds,
                        "freeze_layers": freeze_layers["none"]},
        "t_12_none": {"run_idx": 312, **base_cfg1e4, "dataset": ds,
                         "freeze_layers": freeze_layers["none"]},

        "q_21_none": {"run_idx": 321, **base_cfg5e3, "dataset": ds,
                        "freeze_layers": freeze_layers["none"]},
        "t_22_none": {"run_idx": 322, **base_cfg5e3, "dataset": ds,
                         "freeze_layers": freeze_layers["none"]},

        "q_31_none": {"run_idx": 331, **base_cfg1e3, "dataset": ds,
                        "freeze_layers": freeze_layers["none"]},
        "t_32_none": {"run_idx": 332, **base_cfg1e3, "dataset": ds,
                         "freeze_layers": freeze_layers["none"]},
    }

    for k in train_jobs.keys():
        back_bone_key = k.split("_")[0]
        back_bone_info = back_bone_by_notebook[back_bone_key]

        train_jobs[k]["back_bone"] = {
                "info": back_bone_info, "embedding_model": None, "preprocess_input_layer": None
        }

    return train_jobs


def validate_job_list():
    jobs_list_ = job_list(False)
    non_local_keys = list(filter(lambda x: not x.startswith("l"), jobs_list_.keys()))

    join_ds_name_variante = lambda ds: "_".join([ds["name"], ds["variation"]])
    array_to_str_lst = lambda lst: list(map(lambda x: str(x), lst))
    join_backbone_inf = lambda back_bone: "_".join(array_to_str_lst(back_bone["info"].values()))
    join_job = lambda j: "_".join([join_ds_name_variante(j["dataset"]), join_backbone_inf(j["back_bone"])])

    jobs = map(lambda k: jobs_list_[k], non_local_keys)
    jobs = list(map(join_job, jobs))

    jobs_distinct = distinct(jobs)
    assert len(jobs) != jobs_distinct, "At least one duplicate Job!"

validate_job_list()