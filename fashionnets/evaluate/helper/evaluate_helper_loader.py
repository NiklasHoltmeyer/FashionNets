from fashiondatasets.deepfashion1.DeepFashion1 import DeepFashion1Dataset
from fashionnets.train_jobs.loader.dataset_loader import prepare_ds
from fashionnets.models.layer.Augmentation import compose_augmentations
import matplotlib.pyplot as plt
from fashionnets.train_jobs.loader.checkpoint_loader import load_latest_checkpoint
from fashionnets.models.layer.Augmentation import compose_augmentations
from fashionnets.train_jobs.loader.model_loader import load_siamese_model_from_train_job


def load_model_clean(train_job):
    train_job["back_bone"]["info"]["is_triplet"] = True
    train_job["format"] = 'triplet'
    train_job["is_triplet"] = True

    siamese_model_triplet, init_epoch, _callbacks = load_siamese_model_from_train_job(load_weights=False, **train_job)

    train_job["back_bone"]["info"]["is_triplet"] = False
    train_job["format"] = 'quadruplet'
    train_job["is_triplet"] = False

    siamese_model_quadtruplet, init_epoch, _callbacks = load_siamese_model_from_train_job(load_weights=False,
                                                                                          **train_job)

    return siamese_model_triplet, siamese_model_quadtruplet


def load_model(cp_path, train_job):
    siamese_model_triplet, siamese_model_quadtruplet = load_model_clean(train_job=train_job)

    train_job["path"]["checkpoint"] = cp_path

    (t_succ, t_init_epoch) = load_latest_checkpoint(siamese_model_triplet, **train_job)
    (q_succ, q_init_epoch) = load_latest_checkpoint(siamese_model_quadtruplet, **train_job)

    assert q_succ == t_succ and t_succ
    assert t_init_epoch == q_init_epoch

    print(t_init_epoch - 1, "EPOCH")

    return siamese_model_triplet, siamese_model_quadtruplet


def load_ds(base_path="./deep_fashion_1_256", image_suffix="_256"):
    ds_loader = DeepFashion1Dataset(base_path,
                                    image_suffix=image_suffix,
                                    model=None, nrows=None,
                                    augmentation=compose_augmentations()(False),
                                    generator_type="apn",
                                    embedding_path="./emb",
                                    hard_sampling=None,
                                    batch_size=32,
                                    n_chunks=6)

    t_ds = ds_loader.load(splits=["test", "val"],
                          is_triplet=True,
                          force_train_recreate=False,
                          force=False, force_hard_sampling=False)

    q_ds = ds_loader.load(splits=["test", "val"],
                          is_triplet=False,
                          force_train_recreate=False,
                          force=False, force_hard_sampling=False)

    return t_ds, q_ds