from fashionnets.train_jobs.loader.dataset_loader import loader_info
from fashionnets.train_jobs.settings.default_settings import base_settings, back_bone_settings


def job_list(debugging):
    base_cfg = base_settings(debugging)
    # back_bone_cfg =
    deep_fash_cfg = lambda variation: loader_info("deep_fashion_256", "variation")

    back_bone_by_notebook = {
        "g_i": back_bone_settings("resnet50", weights="imagenet", is_triplet=False),
        "k_ok": back_bone_settings("resnet50", weights=None, is_triplet=False),

        "g_v": back_bone_settings("resnet50", weights="imagenet", is_triplet=True),
        "g_p": back_bone_settings("resnet50", weights=None, is_triplet=True),

        "g_b": back_bone_settings("resnet50", weights="imagenet", is_triplet=True),
        "g_ok": back_bone_settings("resnet50", weights=None, is_triplet=True),

        "l_i1": back_bone_settings("resnet50", weights=None, is_triplet=True),
        "l_i2": back_bone_settings("resnet50", weights="imagenet", is_triplet=True),
        "l_i3": back_bone_settings("simplecnn", weights=None, is_triplet=True),
    }

    train_jobs = {
        "g_i": {"run_idx": 31, **base_cfg, "dataset": deep_fash_cfg("df_quad_3")},
        "k_ok": {"run_idx": 32, **base_cfg, "dataset": deep_fash_cfg("df_quad_3")},
        "g_v": {"run_idx": 33, **base_cfg, "dataset": deep_fash_cfg("df_quad_3")},
        "g_p": {"run_idx": 34, **base_cfg, "dataset": deep_fash_cfg("df_quad_3")},

        "g_b": {"run_idx": 25, **base_cfg, "dataset": deep_fash_cfg("df_quad_2")},
        "g_ok": {"run_idx": 26, **base_cfg, "dataset": deep_fash_cfg("df_quad_2")},

        "l_i1": {"run_idx": 1177, **base_cfg, "dataset": deep_fash_cfg("df_quad_3")},
        "l_i2": {"run_idx": 1188, **base_cfg, "dataset": deep_fash_cfg("df_quad_3")},
        "l_i3": {"run_idx": 1199, **base_cfg, "dataset": deep_fash_cfg("df_quad_3")}
    }

    for k in train_jobs.keys():
        train_jobs[k]["back_bone"] = {
            "info": back_bone_by_notebook[k], "embedding_model": None, "preprocess_input_layer": None
        }

    return train_jobs