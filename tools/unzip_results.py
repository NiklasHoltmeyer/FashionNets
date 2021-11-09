import os
import zipfile
from pathlib import Path

from fashionnets.train_jobs.environment.Environment_Consts import notebooks

results_download_path = notebooks["local"]["paths"]["results_download_path"]

for folder in os.listdir(results_download_path):
    full_path = Path(results_download_path, folder)

    if full_path.is_file() or not "triplet" in folder:
        continue

    header = None
    lines = []

    for zip_file in os.listdir(full_path):
        full_zip = Path(full_path, zip_file)
        unzipped_file = zipfile.ZipFile(full_zip, "r")
        history = unzipped_file.read("history.csv").decode("utf-8")
        history_lines = str(history).split("\n")
        if not header:
            header = history_lines[0]
            lines.append(header)
        value_lines = history_lines[1:]
        value_lines = list(filter(lambda l: len(l) > 0, value_lines))

        lines.append(value_lines)
    print(lines)

# unzipped_file = zipfile.ZipFile("sample.zip", "r")
# a_file = unzipped_file.read("test.txt")
