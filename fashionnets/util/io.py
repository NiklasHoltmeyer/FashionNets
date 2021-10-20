def read_file(path):
    with open(path, "r") as f:
        return f.read()


def write_file(path, data, append=False):
    op = "w+" if not append else "a+"

    with open(path, op) as f:
        f.write(data)


import json


def json_load(file_path):
    with open(file_path, ) as f:
        data = json.load(f)
        return data
