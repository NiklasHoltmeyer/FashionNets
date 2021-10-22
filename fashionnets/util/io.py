import json
import kaggle


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
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(ds_name, path, unzip=unzip)
