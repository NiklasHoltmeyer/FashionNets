import os
from pathlib import Path

from fashionnets.train_jobs.environment.Environment import Environment
from fashionnets.train_jobs.environment.Environment_Consts import kaggle, notebooks
from fashionnets.util.io import json_load

# noinspection PyBroadException
from fashiondatasets.utils.logger.defaultLogger import defaultLogger

try:
    # Imports that only work withing Google Colab
    # noinspection PyPackageRequirements,PyUnresolvedReferences
    from google.colab import drive
except:
    pass

logger = defaultLogger("deepfashion_environment")

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
            # noinspection PyPackageRequirements,PyUnresolvedReferences
            from google.colab import files
            logger.error("Please upload Kaggle.json")
            files.upload()
            json_path = "kaggle.json"
        os.system(f"cp {json_path} ~/.kaggle/")
