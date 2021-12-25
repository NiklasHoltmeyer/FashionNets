import os
from pathlib import Path
import time

from fashionnets.train_jobs.environment.Environment_Builder import EnvironmentBuilder
from fashionnets.train_jobs.environment.Environment_Consts import notebooks

env = EnvironmentBuilder.by_runtime_notebook_name("l_")
env.set_name("$_DEL_ME_$")
env.init(skip_dependencies=True)
client = env.webdav.client
remote_base_path = env.webdav.base_path.replace("$_DEL_ME_$", "").replace("//", "/")

results_download_path = notebooks["local"]["paths"]["results_download_path"]


def download_results(keep_latest_version=True, extension=".zip", only_print=False):
    files = client.list(remote_base_path)
    result_folders = filter(lambda x: "quadruplet" in x or "triplet" in x, files)

    for result_folder in result_folders:
        remote_path = "/".join([remote_base_path, result_folder])
        remote_files = client.list(remote_path)

        files = list(filter(lambda x: x.endswith(extension), remote_files))

        files_full_remote_path = map(
            lambda x: "/".join([remote_base_path, result_folder, x]).replace("//", "/"),
            files)

        target_folder = Path(results_download_path, result_folder).resolve()
        files_full_remote_path = sorted(files_full_remote_path)

        if len(files_full_remote_path) < 2:
            continue

        if keep_latest_version:
            files_full_remote_path = files_full_remote_path[:-1]

        for remote_zip in files_full_remote_path:
            cmd = f'rclone move "hi:/{remote_zip}" "{target_folder}" -P'

            if only_print:
                print(cmd)
            else:
                print(os.system(cmd))

while True:
    download_results(extension=".zip", keep_latest_version=True, only_print=False)
    download_results(extension=".csv", keep_latest_version=True, only_print=False)
    time.sleep(60 * 15)
