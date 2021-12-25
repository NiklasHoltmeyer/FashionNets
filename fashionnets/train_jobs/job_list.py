from fashionnets.models.embedding.resnet50 import ResNet50Builder
from fashionnets.models.layer.Augmentation import compose_augmentations
from fashionnets.train_jobs.loader.dataset_loader import loader_info
from fashionnets.train_jobs.settings.default_settings import base_settings, back_bone_settings
from fashionscrapper.utils.list import distinct


def job_config_by_notebook_name(notebook_name, debugging):
    x = notebook_name.split("_")[1:]
    job_id = "_".join(x)  # g_b1_12 => b1_12
    return job_list(debugging)[job_id]


def job_list(debugging):
    base_cfg1e4 = base_settings(debugging, 1e-4)
    #    base_cfg5e3 = base_settings(debugging, 5e-3)
    #    base_cfg1e3 = base_settings(debugging, 1e-3)
    base_cfg1e5 = base_settings(debugging, 1e-5)

    # back_bone_cfg =
    triplet_w_weights = back_bone_settings("resnet50", weights="imagenet", is_triplet=True)
    #    triplet_no_weights = back_bone_settings("resnet50", weights=None, is_triplet=True)

    quad_w_weights = back_bone_settings("resnet50", weights="imagenet", is_triplet=False)
    #    quad_no_weights = back_bone_settings("resnet50", weights=None, is_triplet=False)

    ds = loader_info("deep_fashion_1")
    ds_own = loader_info("own")

    back_bone_by_notebook = {
        #        "b1": {**quad_no_weights},
        "q": {**quad_w_weights},
        "t": {**triplet_w_weights},
        #        "b4": {**triplet_no_weights},
    }

    freeze_layers = {
        "non_conv5_block1_out": ResNet50Builder.freeze_non_conv5_block1_out,
        "first_30": lambda model: ResNet50Builder.freeze_first_n_layers(model, 30),
        "none": None
    }

    train_jobs = {
        # region Job_List_Old
        # None
        # WARNUNG! noch mit altem loss!        "q_11_none": {"run_idx": 311, **base_cfg1e5, "dataset": ds, "freeze_layers": freeze_layers["none"]},

        "t_apn": {"run_idx": 221, **base_cfg1e5, "dataset": ds,
                  "freeze_layers": freeze_layers["none"],
                  "augmentation": compose_augmentations()},

        "q_apn": {"run_idx": 222, **base_cfg1e5, "dataset": ds,
                  "freeze_layers": freeze_layers["none"],
                  "augmentation": compose_augmentations()},

        "t_ctl": {"run_idx": 231, **base_cfg1e5, "dataset": ds,
                  "freeze_layers": freeze_layers["none"],
                  "augmentation": compose_augmentations(), "generator_type": "ctl",
                  "is_ctl": True},

        "q_ctl": {"run_idx": 242, **base_cfg1e5, "dataset": ds,
                  "freeze_layers": freeze_layers["none"],
                  "augmentation": compose_augmentations(),
                  "generator_type": "ctl", "n1_sample": "centroid",
                  "is_ctl": True},

        "q_ctl_instance": {"run_idx": 252, **base_cfg1e5, "dataset": ds,
                           "freeze_layers": freeze_layers["none"],
                           "augmentation": compose_augmentations(),
                           "generator_type": "ctl", "n1_sample": "instance",
                           "is_ctl": True},

        # endregion
        # region Pre
        "q_pre": {"run_idx": 11, **base_cfg1e5, "dataset": ds_own,
                  "freeze_layers": freeze_layers["none"],
                  "augmentation": compose_augmentations()},

        "q_ctl_pre": {"run_idx": 12, **base_cfg1e5, "dataset": ds_own,
                      "freeze_layers": freeze_layers["none"],
                      "augmentation": compose_augmentations(), "generator_type": "ctl", "n1_sample": "centroid",
                      "is_ctl": True},

        "q_ctl_instance_pre": {"run_idx": 15, **base_cfg1e5, "dataset": ds_own,
                               "freeze_layers": freeze_layers["none"],
                               "augmentation": compose_augmentations(),
                               "generator_type": "ctl", "n1_sample": "instance",
                               "is_ctl": True},

        "t_ctl_pre": {"run_idx": 13, **base_cfg1e5, "dataset": ds_own,
                      "freeze_layers": freeze_layers["none"],
                      "augmentation": compose_augmentations(), "generator_type": "ctl",
                      "is_ctl": True},
        # endregion

        "q_fine_1": {"run_idx": 13371, **base_cfg1e5, "dataset": ds_own,
                  "freeze_layers": freeze_layers["none"],
                  "augmentation": compose_augmentations()},

        "q_fine_2": {"run_idx": 13372, **base_cfg1e5, "dataset": ds_own,
                     "freeze_layers": freeze_layers["non_conv5_block1_out"],
                     "augmentation": compose_augmentations()},

        "q_fine_3": {"run_idx": 13373, **base_cfg1e5, "dataset": ds_own,
                     "freeze_layers": freeze_layers["first_30"],
                     "augmentation": compose_augmentations()},
    }
    #

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
