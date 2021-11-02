from fashionscrapper.utils.list import distinct

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
    deep_fash_cfg = lambda variation: loader_info("deep_fashion_256", variation)

    triplet_w_weights = back_bone_settings("resnet50", weights="imagenet", is_triplet=True)
    triplet_no_weights = back_bone_settings("resnet50", weights=None, is_triplet=True)

    quad_w_weights = back_bone_settings("resnet50", weights="imagenet", is_triplet=False)
    quad_no_weights = back_bone_settings("resnet50", weights=None, is_triplet=False)

    ds = deep_fash_cfg("df_quad_2")

    back_bone_by_notebook = {
        "b1": {**quad_no_weights},
        "b2": {**quad_w_weights},

        "b3": {**triplet_w_weights},
        "b4": {**triplet_no_weights},
    }

    train_jobs = {
#        "b1_11": {"run_idx": 11, **base_cfg1e4, "dataset": ds}, #done
#        "b2_12": {"run_idx": 12, **base_cfg1e4, "dataset": ds}, #
#        "b3_13": {"run_idx": 13, **base_cfg1e4, "dataset": ds}, #läuft
#        "b4_14": {"run_idx": 14, **base_cfg1e4, "dataset": ds}, #

        ###
#        "b1_21": {"run_idx": 21, **base_cfg5e3, "dataset": ds}, #gecanceld nach 3 eps (1.0)
#        "b2_22": {"run_idx": 22, **base_cfg5e3, "dataset": ds}, #läuft
#        "b3_23": {"run_idx": 23, **base_cfg5e3, "dataset": ds}, #läuft
#        "b4_24": {"run_idx": 24, **base_cfg5e3, "dataset": ds}, #Done

        ##
#        "b1_31": {"run_idx": 31, **base_cfg1e3, "dataset": ds}, #done
#        "b2_32": {"run_idx": 32, **base_cfg1e3, "dataset": ds}, # gecanceld nach 9ep (1.0)
#        "b3_33": {"run_idx": 33, **base_cfg1e3, "dataset": ds}, #
#        "b4_34": {"run_idx": 34, **base_cfg1e3, "dataset": ds}, # Läuft

        "b1_111": {"run_idx": 11, **base_cfg1e4, "dataset": ds}, #done
        "b2_112": {"run_idx": 12, **base_cfg1e4, "dataset": ds}, #
        "b3_113": {"run_idx": 13, **base_cfg1e4, "dataset": ds}, #läuft
        "b4_114": {"run_idx": 14, **base_cfg1e4, "dataset": ds}, #

        ###
        "b1_121": {"run_idx": 21, **base_cfg5e3, "dataset": ds}, #gecanceld nach 3 eps (1.0)
        "b2_122": {"run_idx": 22, **base_cfg5e3, "dataset": ds}, #läuft
        "b3_123": {"run_idx": 23, **base_cfg5e3, "dataset": ds}, #läuft
        "b4_124": {"run_idx": 24, **base_cfg5e3, "dataset": ds}, #Done

        ##
        "b1_131": {"run_idx": 31, **base_cfg1e3, "dataset": ds}, #done
        "b2_132": {"run_idx": 32, **base_cfg1e3, "dataset": ds}, # gecanceld nach 9ep (1.0)
        "b3_133": {"run_idx": 33, **base_cfg1e3, "dataset": ds}, #
        "b4_134": {"run_idx": 34, **base_cfg1e3, "dataset": ds}, # Läuft
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