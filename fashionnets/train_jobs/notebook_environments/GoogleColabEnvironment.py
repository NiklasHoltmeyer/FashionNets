import json
import os
from pathlib import Path

from fashionnets.train_jobs.notebook_environments.Environment import Environment
from fashionnets.train_jobs.notebook_environments.consts import kaggle, notebooks
from fashionnets.util.io import json_load

try:
    # Imports that only work withing Google Colab
    from google.colab import drive
except:
    pass


class GoogleColabEnvironment(Environment):
    def __init__(self):
        super(GoogleColabEnvironment, self).__init__("google")

    def load_webdav(self):
        assert self.train_job_name, "Set Run Name via GoogleColabEnv::set_name first"
        return json_load(notebooks["google"]["paths"]["secrets"]["webdav"])

    def load_kaggle(self):
        # Load Secrets
        if Path(kaggle["paths"]["secret"]).exists():
            return

        if not Path("/gdrive/").exists():
            drive.mount('/gdrive/')

        os.system("mkdir ~/.kaggle")

        gdrive_kaggle_path = notebooks["google"]["paths"]["secrets"]["kaggle"]

        if Path(gdrive_kaggle_path).exists():
            json_path = gdrive_kaggle_path
        else:
            from google.colab import files
            print("Please upload Kaggle.json")
            files.upload()
            json_path = "kaggle.json"
        os.system(f"cp {json_path} ~/.kaggle/")
