from pathlib import Path

def read_file(path):
    with open(path, "r") as f:
        return f.read()

def write_file(path, data, append=False):
    op = "w+" if not append else "a+"

    with open(path, op) as f:
        f.write(data)