from fashiondatasets.own.Quadruplets import Quadruplets
from fashiondatasets.own.helper.mappings import preprocess_image
import tensorflow as tf

from fashionnets.models.embedding.simple_cnn import SimpleCNN


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
    train_dataset, val_dataset, n_train, n_val = _load_dataset(**{"base_path":base_path, **ds_settings})
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


def _load_image_preprocessor(format, target_shape, preprocess_input=None, **kwargs):
    is_triplet = "triplet" in format
    prep_image = preprocess_image(target_shape, preprocess_img=preprocess_input)

    if is_triplet:
        return lambda a, p, n: (prep_image(a), prep_image(p), prep_image(n))
    else:
        return lambda a, p, n1, n2: (prep_image(a), prep_image(p), prep_image(n1), prep_image(n2))

def _load_dataset(base_path, batch_size, buffer_size, train_split, format, **settings):
    split = train_split

    quad = Quadruplets(base_path, map_full_paths=True, format=format)

    dataset = quad.load_as_dataset()
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=False)\
        .prefetch(tf.data.AUTOTUNE)

    n_total_items = len(quad)
    n_train_items = round(split * n_total_items) #  // batch_size
    n_val_items = n_total_items - n_train_items

    n_train_batches = n_train_items // batch_size

    train_dataset = dataset.take(n_train_batches).map(_load_image_preprocessor(format, **settings))
    val_dataset = dataset.skip(n_train_batches).map(_load_image_preprocessor(format, **settings))

    print("n_total_items", "\t", n_total_items)
    print("n_train_items", "\t", n_train_items)
    print("n_val_items", "\t", n_val_items)
    print("n_train_batches", " ", n_train_batches)

    return train_dataset, val_dataset, n_train_items, n_val_items

#    n_items = len(dataset)
#    n_train_items = round(n_items * split)

#    train_dataset = dataset.take(n_train_items)
#    val_dataset = dataset.skip(n_train_items)

#    n_val_items = len(val_dataset)

#    train_dataset = train_dataset.batch(batch_size, drop_remainder=False)
#    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE) #.prefetch(tf.data.AUTOTUNE)

#    val_dataset = val_dataset.batch(batch_size, drop_remainder=False)
#    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

#    if settings.get("verbose", False):
#        print("# Items", n_items)
#        print("# Train", n_train_items)
#        print("# Val  ", n_val_items)

#    return train_dataset, val_dataset, n_train_items, n_val_items

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
        dataset_base_path = "/content/own_256" #/content/own_256
    elif notebook == "kaggle":
        dataset_base_path = "/kaggle/working/own_256" #"../input/own-sl-256/own_256"
    else:
        dataset_base_path = "F:\\workspace\\datasets\\own_256"

    return dataset_base_path

if __name__ == "__main__":
    global_settings = {
        "batch_size": 32,
        "notebook": "local",
        "verbose": True,
        #    "nrows": 1,
        "epochs": 25
    }
    def job_3(input_shape):
        alpha = 1.0
        beta = 0.5

        weights = "none"
        back_bone = "simplecnn"

        is_triplet = True

        run_name = f"{back_bone}_{weights}"

        back_bone, preprocess_input = SimpleCNN.build(input_shape), None

        _format = "triplet" if is_triplet else "quadruplet"

        d = load_train_job(run_name, format=_format, preprocess_input=preprocess_input,
                           **global_settings, target_shape=input_shape)

        local_settings = {
            "alpha": alpha, "beta": beta,
            "triplets": is_triplet, "is_triplet": is_triplet,
            "input_shape": input_shape,
            "back_bone": back_bone,
            "preprocess_input": preprocess_input

        }
        return {**global_settings, **d, **local_settings}

    job = job_3((144,144))
    ds = job["run"]["dataset"]
    train, val = ds["train"], ds["val"]

    print("TRAIN DS")
    if list(iter(train.take(1))) is None:
        print("Train Fail")

    print("VAL DS")
    if list(iter(val.take(1))) is None:
        print("Val Fail")
    print(job.keys())
