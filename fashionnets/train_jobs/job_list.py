from fashionscrapper.utils.list import distinct

from fashionnets.train_jobs.loader.dataset_loader import loader_info
from fashionnets.train_jobs.settings.default_settings import base_settings, back_bone_settings


def job_list(debugging):
    base_cfg = base_settings(debugging)
    # back_bone_cfg =
    deep_fash_cfg = lambda variation: loader_info("deep_fashion_256", variation)

    back_bone_by_notebook = {
        "g_i": back_bone_settings("resnet50", weights="imagenet", is_triplet=False),
        "k_ok": back_bone_settings("resnet50", weights=None, is_triplet=False),

        "g_v": back_bone_settings("resnet50", weights="imagenet", is_triplet=True),
        "g_p": back_bone_settings("resnet50", weights=None, is_triplet=True),

        "g_b": back_bone_settings("resnet50", weights="imagenet", is_triplet=True),
        "g_ok": back_bone_settings("resnet50", weights=None, is_triplet=True),

        "g_i2": back_bone_settings("resnet50", weights="imagenet", is_triplet=False),  # ImgNet False V2
        "k_ok2": back_bone_settings("resnet50", weights=None, is_triplet=False),  # None   False V2
        "g_v2": back_bone_settings("resnet50", weights="imagenet", is_triplet=True),  # ImgNet True V1
        "g_p2": back_bone_settings("resnet50", weights=None, is_triplet=True),  # None   True V1

        "k_ok3": back_bone_settings("resnet50", weights="imagenet", is_triplet=False),  # ImgNet False V1
        "g_i3": back_bone_settings("resnet50", weights=None, is_triplet=False),  # None   False V1


        "l_i1": back_bone_settings("resnet50", weights=None, is_triplet=True),
        "l_i2": back_bone_settings("resnet50", weights="imagenet", is_triplet=True),
        "l_i3": back_bone_settings("simplecnn", weights=None, is_triplet=True),
    }

    train_jobs = {
        "g_i": {"run_idx": 31, **base_cfg, "dataset": deep_fash_cfg("df_quad_3")},  # ImgNet False V3     # Done
        "k_ok": {"run_idx": 32, **base_cfg, "dataset": deep_fash_cfg("df_quad_3")},  # None   False V3    # Done
        "g_v": {"run_idx": 33, **base_cfg, "dataset": deep_fash_cfg("df_quad_3")},  # ImgNet True V3
        "g_p": {"run_idx": 34, **base_cfg, "dataset": deep_fash_cfg("df_quad_3")},  # None   True V3

        "g_b": {"run_idx": 25, **base_cfg, "dataset": deep_fash_cfg("df_quad_2")},  # ImgNet True V2
        "g_ok": {"run_idx": 26, **base_cfg, "dataset": deep_fash_cfg("df_quad_2")},  # None   True V2

        "l_i1": {"run_idx": 1177, **base_cfg, "dataset": deep_fash_cfg("df_quad_3")},
        "l_i2": {"run_idx": 1188, **base_cfg, "dataset": deep_fash_cfg("df_quad_3")},
        "l_i3": {"run_idx": 1199, **base_cfg, "dataset": deep_fash_cfg("df_quad_3")},
        ## weiter
        "g_i2": {"run_idx": 21, **base_cfg, "dataset": deep_fash_cfg("df_quad_2")},  # ImgNet False V2  # Done
        "k_ok2": {"run_idx": 22, **base_cfg, "dataset": deep_fash_cfg("df_quad_2")},  # None   False V2  #Done
        "g_v2": {"run_idx": 11, **base_cfg, "dataset": deep_fash_cfg("df_quad_1")},  # ImgNet True V1
        "g_p2": {"run_idx": 12, **base_cfg, "dataset": deep_fash_cfg("df_quad_1")},  # None   True V1

        "k_ok3": {"run_idx": 13, **base_cfg, "dataset": deep_fash_cfg("df_quad_1")},  # ImgNet False V1  #<- auf kaggle
        "g_i3": {"run_idx": 14, **base_cfg, "dataset": deep_fash_cfg("df_quad_1")},  # None   False V1

    }

    for k in train_jobs.keys():
        train_jobs[k]["back_bone"] = {
            "info": back_bone_by_notebook[k], "embedding_model": None, "preprocess_input_layer": None
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
