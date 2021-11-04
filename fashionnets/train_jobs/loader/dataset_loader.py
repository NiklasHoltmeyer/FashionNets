from pathlib import Path

import tensorflow as tf
from fashiondatasets.deepfashion1.DeepFashion1 import DeepFashion1Dataset
from fashiondatasets.deepfashion1.helper.deep_fashion_1_pairs_generator import DeepFashion1PairsGenerator
from fashiondatasets.deepfashion2.DeepFashion2Quadruplets import DeepFashion2Quadruplets
from fashiondatasets.deepfashion2.helper.pairs.deep_fashion_2_pairs_generator import DeepFashion2PairsGenerator
from fashiondatasets.own.Quadruplets import Quadruplets
from fashiondatasets.own.helper.mappings import preprocess_image

from fashionnets.train_jobs.loader.path_loader import _load_dataset_base_path
from fashionnets.util.io import all_paths_exist


def loader_info(name, variation):
    if "deep_fashion_2" in name:
        return deep_fashion_2_loader_info(variation)
    if "deep_fashion_1" in name:
        return deep_fashion_1_loader_info()
    if "own" in name:
        return own_loader_info(variation)
    raise Exception(f"Unknown Loader Information {name} {variation}")


def own_loader_info(variation):
    print("Warning! " * 72)
    print("Dataset Loader Implement own Loader")
    print("#TODO Implement")
    return {
        "name": "own_256",
    }


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
        "variation": "deepfashion1_info",  # "df_quad_3",
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
                "mv img_256 ./deep_fashion_1_256"
            ],
            "check_existence": lambda: all_paths_exist(["./deep_fashion_1_256"])
        }
    }


def load_dataset_loader(**settings):
    ds_name = settings["dataset"]["name"]
    if ds_name == "own" or ds_name == "own_256":
        return lambda: load_own_dataset(**settings)
    if ds_name == "deep_fashion_2_256":
        return lambda: load_deepfashion_2(**settings)
    if "deepfashion1" in ds_name:
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
        print(header_str)
        print("{")
        for k, v in ds_settings.items():
            tab = " " * 8
            k_str = f"{tab}'{k}':"
            v_str = f"'{v}'"
            padding_len = len(header_str) - len(k_str) - len(v_str) - len(tab)
            padding = max(padding_len, 0)
            pad_str = " " * padding
            print(f"{k_str}{pad_str}{v_str}")
        print("}")
        print("*" * len(header_str))


def load_deepfashion_2(**settings):
    print("Load own DeepFashion", settings["batch_size"], "Batch Size")

    ds_settings = _fill_ds_settings(**settings)
    _print_ds_settings(settings.get("verbose", False), **ds_settings)
    base_path = _load_dataset_base_path(**settings)
    datasets = DeepFashion2Quadruplets(base_path=base_path, split_suffix="_256", format=settings["format"],
                                       nrows=settings["nrows"]) \
        .load_as_datasets(validate_paths=False)
    train_ds_info, val_ds_info = datasets["train"], datasets["validation"]

    train_ds, val_ds = train_ds_info["dataset"], val_ds_info["dataset"]

    settings["_dataset"] = settings.pop("dataset")  # <- otherwise kwargs conflict 2x ds

    train_ds, val_ds = prepare_ds(train_ds, **settings), prepare_ds(val_ds, **settings)

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


def load_deepfashion_1(force_train_recreate=False, **settings):
    print("Load own DeepFashion", settings["batch_size"], "Batch Size")

    ds_settings = _fill_ds_settings(**settings)
    _print_ds_settings(settings.get("verbose", False), **ds_settings)
    base_path = _load_dataset_base_path(**settings)

    ds_loader = DeepFashion1Dataset(base_path=base_path, image_suffix="_256", model=None, nrows=settings["nrows"])
    datasets = ds_loader.load(is_triplet=settings["is_triplet"], force_train_recreate=force_train_recreate)

    train_ds_info, val_ds_info = datasets["train"], datasets["validation"]

    train_ds, val_ds = train_ds_info["dataset"], val_ds_info["dataset"]

    settings["_dataset"] = settings.pop("dataset")  # <- otherwise kwargs conflict 2x ds

    train_ds, val_ds = prepare_ds(train_ds, **settings), prepare_ds(val_ds, **settings)

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
    ds_settings = _fill_ds_settings(**settings)
    _print_ds_settings(settings.get("verbose", False), **ds_settings)

    base_path = _load_dataset_base_path(**settings)
    train_dataset, val_dataset, n_train, n_val = _load_own_dataset(**{"base_path": base_path, **ds_settings})
    return {
        "train": train_dataset,
        "val": val_dataset,
        "shape": ds_settings.get("target_shape"),
        "n_items": {
            "total": n_val + n_train,
            "validation": n_val,
            "train": n_train
        }
    }


def _load_own_dataset(base_path, batch_size, buffer_size, train_split, format, **settings):
    print("Load own DS", batch_size, "Batch Size")
    split = train_split
    settings["format"] = format
    settings["batch_size"] = batch_size

    quad = Quadruplets(base_path, **settings)

    dataset = quad.load_as_dataset()
    dataset = dataset.shuffle(buffer_size)

    n_total_items = len(quad)
    n_train_items = round(split * n_total_items)  # // batch_size
    n_val_items = n_total_items - n_train_items

    train_dataset = dataset.take(n_train_items)
    val_dataset = dataset.skip(n_train_items)

    train_dataset, val_dataset = prepare_ds(train_dataset, **settings), prepare_ds(val_dataset, **settings)

    return train_dataset, val_dataset, n_train_items, n_val_items


def prepare_ds(dataset, batch_size, is_triplet, **settings):
    target_shape = settings["input_shape"]
    return dataset.map(_load_image_preprocessor(target_shape=target_shape, is_triplet=is_triplet)) \
        .batch(batch_size, drop_remainder=False) \
        .prefetch(tf.data.AUTOTUNE)


def _load_image_preprocessor(is_triplet, target_shape, preprocess_img=None):
    prep_image = preprocess_image(target_shape, preprocess_img=preprocess_img)
    assert not preprocess_img, "None of the two Datasets needs further Preprocessing!"

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

def build_dataset_hard_pairs_deep_fashion_1(model, job_settings):
    if Path("./deep_fashion_1_256/train.csv").exists():
        Path("./deep_fashion_1_256/train.csv").unlink()

    embedding_model = model.siamese_network.feature_extractor

    ds_loader = DeepFashion1Dataset(base_path="./deep_fashion_1_256", split_suffix="_256", model=embedding_model)
    ds_loader.load_split("train", is_triplet=False, force=True)

    return load_dataset_loader(**job_settings)()

