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

files = client.list(remote_base_path)
result_folders = filter(lambda x: "quadruplet" in x or "triplet" in x, files)

results_download_path = notebooks["local"]["paths"]["results_download_path"]


def download_results():
    for result_folder in result_folders:
        remote_path = "/".join([remote_base_path, result_folder])
        remote_files = client.list(remote_path)
        zips = filter(lambda x: x.endswith(".zip"), remote_files)
        zips_full_remote_path = map(
            lambda x: "/".join([remote_base_path, result_folder, x]).replace("//", "/"),
            zips)

        target_folder = Path(results_download_path, result_folder).resolve()

        for zip in zips_full_remote_path:
            cmd = f'rclone move "hi:/{zip}" "{target_folder}" -P'
            print(os.system(cmd))


while True:
    download_results()
    time.sleep(60 * 30)