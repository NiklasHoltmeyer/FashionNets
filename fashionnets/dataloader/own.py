from fashiondatasets.own.Quadruplets import Quadruplets
from fashiondatasets.own.helper.mappings import preprocess_image


def load_train_job(name, **kwargs):
    path = {
        "checkpoint": _load_checkpoint_path(name, **kwargs),
    }
    run = {
        "name": name,
        "dataset": load_dataset(**kwargs)
    }

    return {
        "path": path,
        "run": run
    }


def load_dataset(**settings):
    ds_settings = {
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
    verbose = settings.get("verbose", False)

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

    base_path = _load_dataset_base_path(**settings)
    dataset, n_items = _load_dataset(base_path, **ds_settings)
    train_dataset, val_dataset = _split_dataset(dataset, n_items, verbose, **ds_settings)
    return {
        "train": train_dataset,
        "val": val_dataset,
        "shape": ds_settings.get("target_shape")
    }


def _split_dataset(dataset, n_items, verbose, **ds_settings):
    split = ds_settings.pop("train_split")
    batch_size = ds_settings["batch_size"]

    train_dataset = dataset.take(round(n_items * split))
    val_dataset = dataset.skip(round(n_items * split))

    train_dataset = train_dataset.batch(batch_size, drop_remainder=False)
    train_dataset = train_dataset.prefetch(8)

    val_dataset = val_dataset.batch(batch_size, drop_remainder=False)
    val_dataset = val_dataset.prefetch(8)

    if verbose:
        print("# Items", n_items)
        print("# Train", round(n_items * 0.8))
        print("# Val  ", n_items - round(n_items * 0.8))

    return train_dataset, val_dataset


def _load_dataset(base_path, **ds_settings):


    quad_helper = Quadruplets(base_path, **ds_settings)
    dataset = quad_helper.load_as_dataset()
    dataset = dataset.shuffle(ds_settings["buffer_size"])
    dataset = dataset.map(_load_image_preprocessor(**ds_settings))

    return dataset, len(quad_helper)


def _load_image_preprocessor(**ds_settings):
    target_shape = ds_settings.get("target_shape")

    is_triplet = "triplet" in ds_settings["format"]

    preprocess_img = ds_settings.get("preprocess_input", None)
    prep_image = preprocess_image(target_shape, preprocess_img=preprocess_img)

    def preprocess_quadruplet(a, p, n1, n2):
        return prep_image(a), prep_image(p), prep_image(n1), prep_image(n2)

    def preprocess_triplet(a, p, n):
        return prep_image(a), prep_image(p), prep_image(n)

    if is_triplet:
        return preprocess_triplet

    return preprocess_quadruplet


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
    notebook = settings["notebook"]
    assert notebook in ["google", "kaggle", "local"]

    if notebook == "google":
        checkpoint_path = "/content/own_256" #/content/own_256
    elif notebook == "kaggle":
        checkpoint_path = "/kaggle/working/own_256" #"../input/own-sl-256/own_256"
    else:
        checkpoint_path = "F:\\workspace\\datasets\\own"

    return checkpoint_path
