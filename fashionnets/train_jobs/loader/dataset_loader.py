import tensorflow as tf
from fashiondatasets.deepfashion2.DeepFashionQuadruplets import DeepFashionQuadruplets
from fashiondatasets.own.Quadruplets import Quadruplets
from fashiondatasets.own.helper.mappings import preprocess_image

from fashionnets.train_jobs.loader.job_loader import _load_dataset_base_path
from fashionnets.util.io import all_paths_exist


def loader_info(name, variation):
    if "deep_fashion" in name:
        return deep_fashion_loader_info(variation)
    raise Exception("TODO")

def deep_fashion_loader_info(variation):
    variation_cmds = "df-quad-2".replace("-", "_")
    return {
        "name": "deep_fashion_256",
        "variation": variation,#"df_quad_3",
        "preprocess": {
            "commands": [
                "mkdir -p ./deep_fashion_256",
                "mv ./train_256 ./deep_fashion_256",
                "mv ./validation_256 ./deep_fashion_256",
                f"mv ./{variation_cmds}/train ./deep_fashion_256",
                f"mv ./{variation_cmds}/validation ./deep_fashion_256",
                f"rmdir ./{variation_cmds}"
            ],
            "check_existence": lambda: all_paths_exist(["./deep_fashion_256"])
        }
    }

def load_dataset(**settings):
    ds_name = settings["dataset"]["name"]
    if ds_name== "own":
        return load_own_dataset(**settings)
    if ds_name == "deep_fashion_256":
        return load_deepfashion(**settings)
    raise Exception(f'Unknown Dataset {ds_name}')


def _fill_ds_settings(**settings):
    return {
            "map_full_paths": settings.get("map_full_paths", True),
            "validate_paths": settings.get("validate_paths", False),
            "format": settings.get("format", "triplet"),  # "quadruplet", # triplet
            "nrows": settings.get("nrows", None),
            "target_shape": settings.get("target_shape", (144, 144)),
            "batch_size": settings.get("batch_size", 32),
            "buffer_size": settings.get("buffer_size", 1024),
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

def load_deepfashion(**settings):
    ds_settings = _fill_ds_settings(**settings)
    _print_ds_settings(settings.get("verbose", False), **ds_settings)
    base_path = _load_dataset_base_path(**settings)
    datasets = DeepFashionQuadruplets(base_path=base_path, split_suffix="_256", format=settings["format"],
                                      nrows=settings["nrows"])\
        .load_as_datasets(validate_paths=False)
    train_ds_info, val_ds_info = datasets["train"], datasets["validation"]

    train_ds, val_ds = train_ds_info["dataset"], val_ds_info["dataset"]

    settings["_dataset"] = settings.pop("dataset") #<- otherwise kwargs conflict 2x ds
    train_ds, val_ds = prepare_ds(train_ds, **settings), prepare_ds(val_ds, **settings)

    n_train, n_val = train_ds_info["n_items"], val_ds_info["n_items"]

    return{
        "train": train_ds,
        "val": val_ds,
        "shape": ds_settings.get("target_shape"),
        "n_items": {
            "total": n_val+n_train,
            "validation": n_val,
            "train": n_train
        }
    }

def load_own_dataset(**settings):
    ds_settings = _fill_ds_settings(**settings)
    _print_ds_settings(settings.get("verbose", False), ds_settings)


    base_path = _load_dataset_base_path(**settings)
    train_dataset, val_dataset, n_train, n_val = _load_own_dataset(**{"base_path":base_path, **ds_settings})
    return {
        "train": train_dataset,
        "val": val_dataset,
        "shape": ds_settings.get("target_shape"),
        "n_items": {
            "total": n_val+n_train,
            "validation": n_val,
            "train": n_train
        }
    }


def _load_own_dataset(base_path, batch_size, buffer_size, train_split, format, **settings):
    split = train_split
    settings["format"] = format

    quad = Quadruplets(base_path, **settings)

    dataset = quad.load_as_dataset()
    dataset = dataset.shuffle(buffer_size)

    n_total_items = len(quad)
    n_train_items = round(split * n_total_items) #  // batch_size
    n_val_items = n_total_items - n_train_items

    train_dataset = dataset.take(n_train_items)
    train_dataset = prepare_ds(train_dataset)

    val_dataset = dataset.skip(n_train_items)
    val_dataset = prepare_ds(val_dataset)

    return train_dataset, val_dataset, n_train_items, n_val_items

def prepare_ds(dataset, batch_size, **settings):
    return dataset.map(_load_image_preprocessor(**settings)) \
        .batch(batch_size, drop_remainder=False) \
        .prefetch(tf.data.AUTOTUNE)

def _load_image_preprocessor(format, target_shape, preprocess_input=None, **kwargs):
    is_triplet = "triplet" in format
    prep_image = preprocess_image(target_shape, preprocess_img=preprocess_input)

    if is_triplet:
        return lambda a, p, n: (prep_image(a), prep_image(p), prep_image(n))
    else:
        return lambda a, p, n1, n2: (prep_image(a), prep_image(p), prep_image(n1), prep_image(n2))
