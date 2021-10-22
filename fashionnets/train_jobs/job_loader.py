import tensorflow as tf
from fashiondatasets.deepfashion2.DeepFashionQuadruplets import DeepFashionQuadruplets
from fashiondatasets.own.Quadruplets import Quadruplets
from fashiondatasets.own.helper.mappings import preprocess_image


def load_dataset(**settings):
    ds_name = settings["dataset"]["name"]
    if ds_name== "own":
        return load_own_dataset(**settings)
    if ds_name == "deep_fashion_256":
        return load_deepfashion(**settings)
    raise Exception(f'Unknown Dataset {ds_name}')

##def _global_settings(notebook, **settings):
##  return {
####      "batch_size": settings.get("batch_size", 32),
####      "buffer_size": settings.get("buffer_size", 32),
####      "notebook": settings.get("notebook", notebook),
####      "verbose": settings.get("verbose", False),
####      "nrows": settings.get("nrows", None),
#####      "epochs": settings.get("epochs", 50),
####      "input_shape": (144, 144)
##  }

def _load_checkpoint_path(run_name, **settings):
    notebook = settings["notebook"]
    assert notebook in ["google", "kaggle", "local"]

    if notebook == "google":
        checkpoint_path = f"/gdrive/MyDrive/results/{run_name}"
    elif notebook == "kaggle":
        checkpoint_path = f"./{run_name}"
    else:
        checkpoint_path = fr"F:\workspace\FashNets\{run_name}"

    return checkpoint_path

def _load_dataset_base_path(**settings):
    #F:\workspace\datasets\own_256
    notebook = settings["notebook"]
    assert notebook in ["google", "kaggle", "local"]

    if notebook == "google":
        dataset_base_path = "/content/" #/content/own_256
    elif notebook == "kaggle":
        dataset_base_path = "/kaggle/working/" #"../input/own-sl-256/own_256"
    else:
        dataset_base_path = "F:\\workspace\\datasets\\"

    return dataset_base_path + settings["dataset"]["name"]


def _load_image_preprocessor(format, target_shape, preprocess_input=None, **kwargs):
    is_triplet = "triplet" in format
    prep_image = preprocess_image(target_shape, preprocess_img=preprocess_input)

    if is_triplet:
        return lambda a, p, n: (prep_image(a), prep_image(p), prep_image(n))
    else:
        return lambda a, p, n1, n2: (prep_image(a), prep_image(p), prep_image(n1), prep_image(n2))


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
    datasets = DeepFashionQuadruplets(base_path=base_path, split_suffix="_256", format=settings["format"])\
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