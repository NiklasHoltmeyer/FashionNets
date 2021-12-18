import logging
import os
from pathlib import Path
from shutil import copyfile

from fashiondatasets.deepfashion1.DeepFashion1 import DeepFashion1Dataset
from fashiondatasets.deepfashion2.DeepFashion2Quadruplets import DeepFashion2Quadruplets
from fashiondatasets.deepfashion2.helper.pairs.deep_fashion_2_pairs_generator import DeepFashion2PairsGenerator
from fashiondatasets.own.Quadruplets import Quadruplets
from fashiondatasets.own.helper.mappings import preprocess_image
import tensorflow as tf
from fashionnets.train_jobs.settings.default_settings import base_settings
from fashionscrapper.utils.io import time_logger

from fashionnets.callbacks.garabe_collector.delete_checkpoints import DeleteOldModel

from fashionnets.models.embedding.resnet50 import EMBEDDING_DIM
from fashionnets.models.embedding.simple_cnn import SimpleCNN
from fashionnets.models.layer.Augmentation import compose_augmentations
from fashionnets.train_jobs.loader.path_loader import _load_dataset_base_path, _load_embedding_base_path, \
    _load_centroid_base_path
from fashionnets.util.io import all_paths_exist
import numpy as np
from fashiondatasets.utils.logger.defaultLogger import defaultLogger

logger = defaultLogger("deepfashion_data_builder", level=logging.INFO)


def loader_info(name, variation=""):
    if "deep_fashion_2" in name:
        return deep_fashion_2_loader_info(variation)
    if "deep_fashion_1" in name:
        return deep_fashion_1_loader_info()
    if "own" in name:
        return own_loader_info()
    raise Exception(f"Unknown Loader Information {name} {variation}")


def own_loader_info():
    return {
        "name": "own_256",
        "variation": "own_256",  # "df_quad_3",
        "preprocess": {
            "commands": [
                "mkdir -p ./own_256",
                "mv asos ./own_256",
                "mv hm ./own_256",
                "mv mango ./own_256",
                "mv entries.json ./own_256",
                "mv entries_test.json ./own_256",
                "mv entries_train.json ./own_256",
                "mv entries_validation.json ./own_256",
                "mv quadruplet.csv ./own_256",
                "mv test.csv ./own_256",
                "mv validation.csv ./own_256",
            ],
            "check_existence": lambda: all_paths_exist(["./own_256"])
        }
    }


#    print("Warning! " * 72)
#    print("Dataset Loader Implement own Loader")
#    print("#TODO Implement")
#    return {
#        "name": "own_256",
#    }


def deep_fashion_2_loader_info(variation):
    variation_commands = variation.replace("-", "_")
    return {
        "name": "deep_fashion_256",
        "variation": variation,  # "df_quad_3",
        "preprocess": {
            "commands": [
                "mkdir -p ./deep_fashion_256",
                "mv ./train_256 ./deep_fashion_256",
                "mv ./validation_256 ./deep_fashion_256",
                f"mv ./{variation_commands}/train ./deep_fashion_256",
                f"mv ./{variation_commands}/validation ./deep_fashion_256",
                f"rmdir ./{variation_commands}"
            ],
            "check_existence": lambda: all_paths_exist(["./deep_fashion_256"])
        }
    }


def deep_fashion_1_loader_info():
    return {
        "name": "deep_fashion_1_256",
        "variation": "deepfashion1info",  # "df_quad_3",
        "preprocess": {
            "commands": [
                "mkdir -p ./deep_fashion_1_256",
                "mv splits.json ./deep_fashion_1_256",
                "mv val.csv ./deep_fashion_1_256",
                "mv cat_idx_by_name.json ./deep_fashion_1_256",
                "mv ids_by_cat_idx.json ./deep_fashion_1_256",
                "mv test.csv ./deep_fashion_1_256",
                "mv cat_name_by_idxs.json ./deep_fashion_1_256",
                "mv train.csv ./deep_fashion_1_256",
                "mv README.txt ./deep_fashion_1_256",
                "mv img_256 ./deep_fashion_1_256",
                "mv ./deepfashion1-info/* ./deep_fashion_1_256",
                "rm -rf ./deepfashion1-info"
            ],
            "check_existence": lambda: all_paths_exist(["./deep_fashion_1_256"])
        }
    }


def load_dataset_loader(**settings):
    ds_name = settings["dataset"]["name"]
    if ds_name == "own" or ds_name == "own_256":
        settings["dataset_hard_pairs_fn"] = build_dataset_hard_pairs_own
        return lambda: load_own_dataset(**settings)
    if ds_name == "deep_fashion_2_256":
        settings["dataset_hard_pairs_fn"] = build_dataset_hard_pairs_deep_fashion_2
        return lambda: load_deepfashion_2(**settings)
    if "deep_fashion_1_256" in ds_name:
        settings["dataset_hard_pairs_fn"] = build_dataset_hard_pairs_deep_fashion_1
        return lambda: load_deepfashion_1(**settings)
    raise Exception(f'Unknown Dataset {ds_name}')


def _fill_ds_settings(**settings):
    keys = ["format", "nrows", "target_shape", "batch_size", "buffer_size"]
    missing_keys = list(filter(lambda k: k not in settings.keys(), keys))
    assert len(missing_keys) == 0, f"Missing Keys: {missing_keys}"

    return {
        "map_full_paths": settings.get("map_full_paths", True),
        "validate_paths": settings.get("validate_paths", False),
        #        "format": settings.get("format", "triplet"),  # "quadruplet", # triplet
        #        "nrows": settings.get("nrows", None),
        #        "target_shape": settings.get("target_shape", (144, 144)),
        #        "batch_size": settings.get("batch_size", 32),
        #        "buffer_size": settings.get("buffer_size", 1024),
        "train_split": settings.get("train_split", 0.8),
        "preprocess_input": settings.get("preprocess_input", None)
    }


def _print_ds_settings(verbose, **ds_settings):
    if verbose:
        header_str = "*" * 24 + " Settings " + "*" * 24
        logger.debug(header_str)
        logger.debug("{")
        for k, v in ds_settings.items():
            tab = " " * 8
            k_str = f"{tab}'{k}':"
            v_str = f"'{v}'"
            padding_len = len(header_str) - len(k_str) - len(v_str) - len(tab)
            padding = max(padding_len, 0)
            pad_str = " " * padding
            logger.debug(f"{k_str}{pad_str}{v_str}")
        logger.debug("}")
        logger.debug("*" * len(header_str))


def load_deepfashion_2(**settings):
    logger.debug(f"Load own DeepFashion {settings['batch_size']} Batch Size")

    ds_settings = _fill_ds_settings(**settings)
    _print_ds_settings(settings.get("verbose", False), **ds_settings)
    base_path = _load_dataset_base_path(**settings)
    datasets = DeepFashion2Quadruplets(base_path=base_path, split_suffix="_256", format=settings["format"],
                                       nrows=settings["nrows"]) \
        .load_as_datasets(validate_paths=False)
    train_ds_info, val_ds_info = datasets["train"], datasets["validation"]

    train_ds, val_ds = train_ds_info["dataset"], val_ds_info["dataset"]

    settings["_dataset"] = settings.pop("dataset")  # <- otherwise kwargs conflict 2x ds

    train_ds, val_ds = prepare_ds(train_ds, is_train=True, **settings), prepare_ds(val_ds, is_train=False, **settings)

    n_train, n_val = train_ds_info["n_items"], val_ds_info["n_items"]

    return {
        "type": "deepfashion_2",
        "train": train_ds,
        "val": val_ds,
        "shape": ds_settings.get("target_shape"),
        "n_items": {
            "total": n_val + n_train,
            "validation": n_val,
            "train": n_train
        }
    }


@time_logger(name="DS-Loader::Load", header="Dataset-Loader", padding_length=50,
             logger=defaultLogger("fashiondataset_time_logger"), log_debug=False)
def load_deepfashion_1(**settings):
    logger.debug(f"Load own DeepFashion {settings['batch_size']} Batch Size")

    ds_settings = _fill_ds_settings(**settings)
    _print_ds_settings(settings.get("verbose", False), **ds_settings)
    base_path = _load_dataset_base_path(**settings)
    embedding_base_path = _load_embedding_base_path(**settings)

    if settings["is_ctl"] or settings["sampling"] == "hard":
        model = settings["back_bone"]["embedding_model"]
        assert model is not None
    else:
        model = None

    # back_bone

    dataframes = settings.get("dataframes", None)

    ds_loader = DeepFashion1Dataset(base_path=base_path,
                                    image_suffix="_256",
                                    model=model,
                                    nrows=settings["nrows"],
                                    augmentation=compose_augmentations()(False),
                                    generator_type=settings["generator_type"],
                                    embedding_path=embedding_base_path,
                                    batch_size=settings["batch_size"],
                                    skip_build=dataframes is not None)

    datasets = ds_loader.load(splits=["train", "val"],
                              is_triplet=settings["is_triplet"],
                              force=settings.get("ds_load_force", False),
                              force_hard_sampling=settings["sampling"] == "hard",
                              embedding_path=embedding_base_path,
                              nrows=settings["nrows"],
                              dataframes=dataframes)

    train_ds_info, val_ds_info = datasets["train"], datasets["validation"]

    train_ds, val_ds = train_ds_info["dataset"], val_ds_info["dataset"]

    settings["_dataset"] = settings.pop("dataset")  # <- otherwise kwargs conflict 2x ds
    train_ds, val_ds = prepare_ds(train_ds, is_train=True, **settings), prepare_ds(val_ds, is_train=False, **settings)
    n_train, n_val = train_ds_info["n_items"], val_ds_info["n_items"]

    return {
        "type": "deepfashion_1",
        "train": train_ds,
        "val": val_ds,
        "shape": ds_settings.get("target_shape"),
        "n_items": {
            "total": n_val + n_train,
            "validation": n_val,
            "train": n_train
        }
    }


def load_own_dataset(**settings):
    train_df, val_df, n_train_items, n_val_items = _load_own_dataset(load_df=True, **settings)

    return load_deepfashion_1(dataframes=[train_df, val_df], **settings)

    # ds_settings = _fill_ds_settings(**settings)
    # _print_ds_settings(**settings)

    # train_dataset, val_dataset, n_train, n_val = _load_own_dataset(**settings)

    # return {


#        "train": train_dataset,
#        "val": val_dataset,
#        "shape": ds_settings.get("target_shape"),
#        "n_items": {
#            "total": n_val + n_train,
#            "validation": n_val,
#            "train": n_train
#        }
#    }


def _load_own_dataset(**settings):
    # logger.debug(f"Load own DS {batch_size} Batch Size")
    # split = train_split
    # settings["format"] = format
    # settings["batch_size"] = batch_size

    base_path = _load_dataset_base_path(**settings)

    quad = Quadruplets(base_path=base_path, split=None, map_full_paths=True, **settings)

    if settings.get("load_df", False):
        train_dataset = quad.load_as_df(split="train", **settings)  # "train
        val_dataset = quad.load_as_df(split="validation", **settings)

        n_train_items, n_val_items = len(train_dataset), len(val_dataset)

        return train_dataset, val_dataset, n_train_items, n_val_items

    n_train_items, train_dataset = quad.load_as_dataset(split="train")  # "train
    n_val_items, val_dataset = quad.load_as_dataset(split="validation")

    settings["_dataset"] = settings.pop("dataset")

    train_dataset, val_dataset = prepare_ds(train_dataset, is_train=True, **settings), \
                                 prepare_ds(val_dataset, is_train=False, **settings)

    return train_dataset, val_dataset, n_train_items, n_val_items


def filter_ds_not_nan(x):
    return not tf.reduce_any(tf.reduce_any(tf.math.is_nan(x)))


def prepare_ds(dataset, batch_size, is_triplet, is_train, **settings):
    target_shape = settings["input_shape"]

    augmentation = settings.get("augmentation", None)

    if augmentation:
        augmentation = augmentation(is_train)

    if settings.get("verbose", False):
        logger.debug(f"Augmentation {augmentation}, IS_Train {is_train}")

    n1_sample = settings.get("n1_sample", None)
    generator_type = settings["generator_type"]

    if generator_type == "ctl":
        assert n1_sample, "n1_sample need to be set if ctl is used"

    return dataset.map(_load_image_preprocessor(target_shape=target_shape, is_triplet=is_triplet,
                                                augmentation=augmentation, generator_type=settings["generator_type"],
                                                n1_sample=n1_sample)) \
        .batch(batch_size, drop_remainder=False) \
        .prefetch(tf.data.AUTOTUNE)


def np_load(feature_path):
    return np.load(feature_path).astype(np.float64)


def load_npy(p):
    d = tf.numpy_function(np_load, [p], tf.float64)
    return tf.convert_to_tensor(d, dtype=tf.float64)


def _load_image_preprocessor(is_triplet, target_shape, generator_type, n1_sample=None, preprocess_img=None,
                             augmentation=None):
    prep_image = preprocess_image(target_shape, preprocess_img=preprocess_img, augmentation=augmentation)
    assert not preprocess_img, "None of the two Datasets needs further Preprocessing!"

    if "ctl" == generator_type:
        if is_triplet:  # Tripl -> A::jpg_path Cp::npy_path Cn::npy_path
            return lambda a, p_ctl, n_ctl: (prep_image(a),
                                            load_npy(p_ctl),
                                            load_npy(n_ctl))
        else:  # Quad -> A::jpg_path N1::npy_path Cp::npy_path C_n1::npy_path C_n2::npy_path
            if n1_sample == "centroid":
                return lambda a, n1, p_ctl, n1_ctl, n2_ctl: (prep_image(a),
                                                             load_npy(p_ctl),
                                                             load_npy(n1_ctl),
                                                             load_npy(n2_ctl),
                                                             )
            if n1_sample == "instance":
                print("N1 Sample Instance"  * 25)
                return lambda a, n1, p_ctl, n1_ctl, n2_ctl: (prep_image(a),
                                                             load_npy(p_ctl),
                                                             load_npy(n1),
                                                             load_npy(n2_ctl),
                                                             )

    if "apn" == generator_type:
        if is_triplet:
            return lambda a, p, n: (prep_image(a), prep_image(p), prep_image(n))
        else:
            return lambda a, p, n1, n2: (prep_image(a), prep_image(p), prep_image(n1), prep_image(n2))


def build_dataset_hard_pairs_deep_fashion_2(model, job_settings):
    if Path("./deep_fashion_256/train/quadruplets.csv").exists():
        Path("./deep_fashion_256/train/quadruplets.csv").unlink()
    embedding_model = model.siamese_network.feature_extractor

    generator = DeepFashion2PairsGenerator(r"./deep_fashion_256",
                                           embedding_model,
                                           split_suffix="_256",
                                           number_possibilites=256)

    apnn = generator.build_anchor_positive_negative1_negative2("train")
    DeepFashion2PairsGenerator.save_pairs_to_csv(generator.base_path, "train", apnn)

    return load_dataset_loader(**job_settings)()


def build_dataset_hard_pairs_deep_fashion_1(model, job_settings, init_epoch, build_frequency):
    job_settings["force_ctl"] = init_epoch > 0

    if init_epoch == 0 and not job_settings["is_ctl"] and not job_settings["sampling"] == "hard":
        return load_dataset_loader(**job_settings)()

    result = __download_hard_pairs(job_settings, init_epoch, build_frequency)

    if result is not None:
        return result

    if (init_epoch % build_frequency) == 0:
        return __build_move_deepfashion_hard_pairs(model, job_settings, init_epoch)

    raise Exception("Could not Download Train.csv.")


def build_dataset_hard_pairs_own(model, job_settings, init_epoch, build_frequency):
    job_settings["force_ctl"] = init_epoch > 0

    if init_epoch == 0 and not job_settings["is_ctl"] and not job_settings["sampling"] == "hard":
        return load_dataset_loader(**job_settings)()

    result = __download_hard_pairs(job_settings,
                                   init_epoch,
                                   build_frequency,
                                   ds_name="own_256")

    if result is not None:
        return result

    if (init_epoch % build_frequency) == 0:
        return __build_move_deepfashion_hard_pairs(model, job_settings, init_epoch, ds_name="own_256")

    raise Exception("Could not Download Train.csv.")


def __build_move_deepfashion_hard_pairs(model, job_settings, init_epoch, ds_name="deep_fashion_1_256"):
    if Path(f"./{ds_name}/train.csv").exists():
        Path(f"./{ds_name}/train.csv").unlink()

        # train_ctl, val_ctl

    #    if model:
    #        embedding_model = model.siamese_network.feature_extractor
    #    else:
    #        embedding_model = None

    embedding_base_path = _load_embedding_base_path(**job_settings) if job_settings["is_ctl"] or \
                                                                       job_settings["sampling"] == "hard" else None

    if embedding_base_path and job_settings["force_ctl"] or job_settings["sampling"] == "hard":
        DeleteOldModel.delete_path(embedding_base_path)
        DeleteOldModel.delete_path(_load_centroid_base_path(**job_settings))

    job_settings["ds_load_force"] = True

    datasets = load_dataset_loader(**job_settings)()

    src = f"./{ds_name}/train.csv"
    dst = f"./{ds_name}/train_{init_epoch:04d}.csv"

    copyfile(src, dst)

    result_uploader = job_settings["environment"].webdav
    result_uploader.move(dst, _async=False)

    return datasets


def __download_hard_pairs(job_settings, init_epoch, build_frequency, ds_name="deep_fashion_1_256"):
    if Path(f"./{ds_name}/train.csv").exists():
        Path(f"./{ds_name}/train.csv").unlink()

    last_epoch = total_epochs(init_epoch, build_frequency) - build_frequency

    dst_name = f"train_{last_epoch:04d}.csv"

    remote = job_settings["environment"].webdav
    csv = filter(lambda d: dst_name in d, remote.list(remote.base_path))
    csv = list(csv)

    if not len(csv) == 1:
        return None

    csv = csv[0]
    csv_path = os.path.join(remote.base_path, csv)

    _callback = lambda: logger.info(f"{csv} downloaded!")

    remote.download(csv_path, f"./{ds_name}/train.csv", callback=_callback, _async=False)

    return load_dataset_loader(**job_settings)()


def total_epochs(init_epoch, build_frequency):
    max_epochs = init_epoch + build_frequency
    if max_epochs % build_frequency == 0:
        return max_epochs
    dif = max_epochs % build_frequency
    return max_epochs - dif
