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




