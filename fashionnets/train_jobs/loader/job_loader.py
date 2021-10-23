from fashionnets.train_jobs.notebook_environments.consts import notebooks


def _load_checkpoint_path(run_name, **settings):
    notebook = settings["notebook"]
    assert notebook in notebooks.keys()

    checkpoint_path = notebooks[notebook]["paths"]["checkpoint"] + run_name

    return checkpoint_path

def _load_dataset_base_path(**settings):
    #F:\workspace\datasets\own_256
    notebook = settings["notebook"]
    assert notebook in notebooks.keys()

    return notebooks[notebook]["paths"]["dataset_base_path"] + settings["dataset"]["name"]


