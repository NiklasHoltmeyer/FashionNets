from fashiondatasets.utils.logger.defaultLogger import defaultLogger
from fashiondatasets.deepfashion1.DeepFashion1 import DeepFashion1Dataset
from fashionnets.train_jobs.loader.dataset_loader import prepare_ds
from fashionnets.train_jobs.loader.checkpoint_loader import load_latest_checkpoint
from fashionnets.models.layer.Augmentation import compose_augmentations
from fashionnets.train_jobs.loader.job_loader import prepare_environment, load_job_settings, add_back_bone_to_train_job
from fashionnets.train_jobs.loader.model_loader import load_siamese_model_from_train_job
from fashiondatasets.utils.logger.defaultLogger import defaultLogger
from fashionnets.train_jobs.loader.path_loader import _load_embedding_base_path, _load_dataset_base_path

logger = defaultLogger()


def load_model_clean(train_job):
    logger.info("Load Siamese Model (Triplet, Quadtruplet)")

    train_job["back_bone"]["info"]["is_triplet"] = True
    train_job["format"] = 'triplet'
    train_job["is_triplet"] = True
    train_job["generator_type"] = "apn"
    train_job["is_ctl"] = False

    siamese_model_triplet, init_epoch, _callbacks = load_siamese_model_from_train_job(load_weights=False, **train_job)
    train_job["generator_type"] = "ctl"
    train_job["is_ctl"] = True

    siamese_model_triplet_ctl, init_epoch, _callbacks = load_siamese_model_from_train_job(load_weights=False,
                                                                                          **train_job)

    logger.info("Load Siamese Model (CTL Triplet, Quadtruplet)")
    train_job["back_bone"]["info"]["is_triplet"] = False
    train_job["format"] = 'quadruplet'
    train_job["is_triplet"] = False
    train_job["generator_type"] = "apn"
    train_job["is_ctl"] = False

    siamese_model_quadtruplet, init_epoch, _callbacks = load_siamese_model_from_train_job(load_weights=False,
                                                                                          **train_job)
    train_job["generator_type"] = "ctl"
    train_job["is_ctl"] = True

    siamese_model_quadtruplet_ctl, init_epoch, _callbacks = load_siamese_model_from_train_job(load_weights=False,
                                                                                              **train_job)

    return siamese_model_triplet, siamese_model_quadtruplet, siamese_model_triplet_ctl, siamese_model_quadtruplet_ctl


def load_model(cp_path, train_job):
    t_model, q_model, tctl_model, qctl_model = load_model_clean(train_job=train_job)

    train_job["path"]["checkpoint"] = cp_path

    (t_succ, t_init_epoch) = load_latest_checkpoint(t_model, **train_job)
    (q_succ, q_init_epoch) = load_latest_checkpoint(q_model, **train_job)

    (tctl_succ, tctl_init_epoch) = load_latest_checkpoint(tctl_model, **train_job)
    (qctl_succ, qctl_init_epoch) = load_latest_checkpoint(qctl_model, **train_job)

    assert q_succ == t_succ and t_succ
    assert t_init_epoch == q_init_epoch

    assert qctl_succ == tctl_succ and tctl_succ
    assert tctl_init_epoch == qctl_init_epoch

    assert t_init_epoch == qctl_init_epoch

    print(t_init_epoch - 1, "EPOCH")

    return {
        "triplet": t_model,
        "quadtruplet": q_model,
        "triplet_ctl": tctl_model,
        "quadtruplet_ctl": qctl_model,
    }

def prepare_dataset(datasets_, job_settings, is_triplet, is_ctl):
    logger.info(f"Prepare DS {is_triplet}, {is_ctl}")
    test_ds_info, val_ds_info = datasets_["test"], datasets_["validation"]
    test_ds, val_ds = test_ds_info["dataset"], val_ds_info["dataset"]

    prepare_settings_ = {
        "is_triplet": is_triplet,
        "generator_type": "apn" if not is_ctl else "ctl",
        "is_ctl": is_ctl,
        "n1_sample": "centroid"
    }

    logger.info(prepare_settings_)

    prepare_settings = {
        "input_shape": job_settings["input_shape"],
        "batch_size": 32,
        "augmentation": compose_augmentations(),
    }
    prepare_settings.update(prepare_settings_)

    test_ds, val_ds = prepare_ds(test_ds, is_train=False, **prepare_settings), prepare_ds(val_ds,
                                                                                          is_train=False,
                                                                                          **prepare_settings)
    test_val_ds = test_ds.concatenate(val_ds)

    return {
        "test": test_ds,
        "validation": val_ds,
        "test+validation": test_val_ds
    }


def load_dataset(model, job_settings):
    base_path = _load_dataset_base_path(**job_settings)
    embedding_base_path = _load_embedding_base_path(**job_settings)

    def __load(model_, generator_type):
        ds_loader = DeepFashion1Dataset(base_path=base_path,
                                    image_suffix="_256",
                                    model=model_,
                                    nrows=job_settings["nrows"],
                                    augmentation=compose_augmentations()(False),
                                    generator_type=generator_type,
                                    embedding_path=embedding_base_path,
                                    batch_size=job_settings["batch_size"])

        job_settings["sampling"] = "random"
        job_settings["ds_load_force"] = False

        t_ds = ds_loader.load(splits=["test", "val"],
                              is_triplet=True,
                              force=job_settings.get("ds_load_force", False),
                              force_hard_sampling=job_settings["sampling"] == "hard",
                              embedding_path=embedding_base_path,
                              nrows=job_settings["nrows"])

        q_ds = ds_loader.load(splits=["test", "val"],
                              is_triplet=False,
                              force=job_settings.get("ds_load_force", False),
                              force_hard_sampling=job_settings["sampling"] == "hard",
                              embedding_path=embedding_base_path,
                              nrows=job_settings["nrows"])

        return t_ds, q_ds

    logger.info("Load DS (Q, T)")
    t_ds, q_ds = __load(None, generator_type="apn")
    logger.info("Load DS (Q, T CTL)")
    t_ctl_ds, q_ctl_ds = __load(model, generator_type="ctl")

    return {
        "triplet": prepare_dataset(t_ds, job_settings, is_triplet=True, is_ctl=False),
        "quadtruplet": prepare_dataset(q_ds, job_settings, is_triplet=False, is_ctl=False),
        "triplet_ctl": prepare_dataset(t_ctl_ds, job_settings, is_triplet=True, is_ctl=True),
        "quadtruplet_ctl": prepare_dataset(q_ctl_ds, job_settings, is_triplet=False, is_ctl=True),
    }

if __name__ == "__main__":
    notebook_name = "l_t_test_ctl"  # 212 t_test_ctl
    environment, training_job_cfg = prepare_environment(notebook_name, debugging=True)

    train_job = load_job_settings(environment=environment, training_job_cfg=training_job_cfg, kaggle_downloader=None)
    job_settings = add_back_bone_to_train_job(environment=environment, **training_job_cfg)

    #modelle = load_model("./", train_job)
    #back_bone_modell = modelle["quadtruplet"].back_bone

    datasets = load_dataset(model=None)


