from fashionscrapper.utils.list import distinct

from fashionnets.train_jobs.loader.dataset_loader import loader_info
from fashionnets.train_jobs.settings.default_settings import base_settings, back_bone_settings


def job_list(debugging):
    base_cfg = base_settings(debugging)
    # back_bone_cfg =
    deep_fash_cfg = lambda variation: loader_info("deep_fashion_256", variation)

    triplet_w_weights = back_bone_settings("resnet50", weights="imagenet", is_triplet=True)
    triplet_no_weights = back_bone_settings("resnet50", weights=None, is_triplet=True)

    quad_w_weights = back_bone_settings("resnet50", weights="imagenet", is_triplet=False)
    quad_no_weights = back_bone_settings("resnet50", weights=None, is_triplet=False)

    ds = deep_fash_cfg("df_quad_2")

    back_bone_by_notebook = {
        "k_ok": {**quad_no_weights},
        "g_i":  {**quad_w_weights},

        "g_ok": {**triplet_w_weights},
        "g_b":  {**triplet_no_weights},

        "g_p_resume_gok": {**triplet_w_weights},
        "g_v_remose_gb": {**triplet_no_weights},

    }

    train_jobs = {
        "k_ok":             {"run_idx": 1,  **base_cfg, "dataset": ds},
        "g_i":              {"run_idx": 2,  **base_cfg, "dataset": ds},

        "g_ok":             {"run_idx": 31, **base_cfg, "dataset": ds},
        "g_b":              {"run_idx": 41, **base_cfg, "dataset": ds},

        "g_p_resume_gok":   {"run_idx": 32, **base_cfg, "dataset": ds},
        "g_v_remose_gb":    {"run_idx": 42, **base_cfg, "dataset": ds},
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
