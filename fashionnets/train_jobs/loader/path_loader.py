from pathlib import Path

from fashionnets.train_jobs.environment.Environment_Consts import notebooks


def _load_checkpoint_path(run_name, **settings):
    notebook = settings["notebook"]
    assert notebook in notebooks.keys()

    checkpoint_path = notebooks[notebook]["paths"]["checkpoint"] + run_name

    return checkpoint_path


def _load_dataset_base_path(**settings):
    # F:\workspace\datasets\own_256
    notebook = settings["notebook"]
    assert notebook in notebooks.keys()

    return notebooks[notebook]["paths"]["dataset_base_path"] + settings["dataset"]["name"]

def _load_embedding_base_path(**settings):
    p = Path(_load_dataset_base_path(**settings), "embeddings")
    return str(p.resolve())

def _load_centroid_base_path(**settings):
    p = Path(_load_dataset_base_path(**settings), "centroids")
    return str(p.resolve())

