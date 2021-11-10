import os
import zipfile
from pathlib import Path


def extract_history(path):
    h_path = Path(path).parent / "history_files"
    h_path.mkdir(exist_ok=True, parents=True)

    with zipfile.ZipFile(path, 'r') as archive:
        history_files = list(filter(lambda d: "history" in d and d.endswith(".pkl"), archive.namelist()))
        for h_file in history_files:
            archive.extract(h_file, h_path)


def extract_run_histories(path):
    for zip_file_name in filter(lambda p: p.endswith(".zip"), os.listdir(path)):
        zip_path = os.path.join(path, zip_file_name)
        extract_history(zip_path)