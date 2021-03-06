import json
from pathlib import Path

from fashiondatasets.utils.logger.defaultLogger import defaultLogger


def read_file(path, flag="r"):
    with open(path, flag) as f:
        return f.read()


def write_file(path, data, append=False):
    op = "w+" if not append else "a+"

    with open(path, op) as f:
        f.write(data)


def json_dump(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


def json_load(file_path):
    with open(file_path, ) as f:
        data = json.load(f)
        return data


def download_extract_kaggle(ds_name, path="./", unzip=True):
    # noinspection PyBroadException
    logger = defaultLogger("deepfashion_environment")
    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(ds_name, path, unzip=unzip)
    except:
        logger.error("Could not find kaggle.json. "
                     "Make sure it's located in /root/.kaggle. Or use the environment method.")


def all_paths_exist(lst):
    path_exist = lambda p: Path(p).exists()
    return all(map(path_exist, lst))


def string_serializer(obj):
    """
    Helper Function to Serialize Objects. Keeps List / Dict Structure, but will Convert everything else to String.
    """
    if type(obj) in [list, tuple]:
        return list(map(string_serializer, obj))
    if type(obj) == dict:
        copy = {}
        for k, v in obj.items():
            copy[k] = string_serializer(v)
        return copy
    return str(obj)
