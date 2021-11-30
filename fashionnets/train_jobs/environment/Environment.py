import logging
import os

from fashiondatasets.utils.logger.defaultLogger import defaultLogger
from kaggle.rest import ApiException

from fashionnets.train_jobs.environment.Environment_Consts import kaggle
from fashionnets.util.remote import WebDav

"""Helper Classes to Init Training-Environment (Setting up Secrets, Defining Env Based Settings ...)
"""

logger = defaultLogger("deepfashion_environment", level=logging.INFO)

# noinspection PyBroadException,PyTypeChecker
class Environment:
    def __init__(self, notebook, download_dependencies=True, dataset_prefix="masterokay/"):
        self.notebook = notebook
        self.train_job_name = None
        self.webdav = None
        self.dependencies = {}
        self.download_dependencies = download_dependencies
        self.dataset_prefix = dataset_prefix

        logger.debug(f"Environment: {self}")

    def set_name(self, name):
        self.train_job_name = name

    def load_webdav(self):
        pass

    def load_kaggle(self):
        pass

    def prepare(self, skip_preprocess):
        preprocess_settings = self.dependencies["kaggle"].get("preprocess", None)

        if not preprocess_settings:
            logger.debug("Preprocess not set")
            return

        if skip_preprocess:
            logger.debug("Skip Preprocess set")
            return

        for cmd in self.dependencies["kaggle"]["preprocess"]["commands"]:
            try:
                os.system(cmd)
            except:
                logger.error(f"Exception {cmd}")
                pass

    def load_dependencies(self, kaggle_downloader=None, skip_preprocess=False):
        if not self.download_dependencies:
            return

        paths_exist = self.dependencies["kaggle"].get("preprocess", {}).get("check_existence", lambda: False)

        if paths_exist():
            logger.debug("All Paths already exist!")
            return

        if kaggle_downloader:
            for ds_name in self.dependencies["kaggle"].values():
                if type(ds_name) is not str:
                    continue
                ds_full_name = kaggle["datasets"]["prefix"] + ds_name.replace("_", "-")
                logger.debug(f"Download: {ds_full_name}")
                try:
                    kaggle_downloader(ds_full_name, f"./", unzip=True)
                except ApiException as e:
                    logger.error(f"Exception: {str(e)}")
                    logger.error(f"ds_full_name: {ds_full_name}")

            self.prepare(skip_preprocess)

            # preprocess
            # operations

    def build_webdav_settings(self, webdav_secrets):
        if not webdav_secrets["base_path"].endswith("/"):
            webdav_secrets["base_path"] += "/"

        webdav_secrets["base_path"] += self.train_job_name

        self.webdav = WebDav(**webdav_secrets)

    def init(self, skip_dependencies=False):
        self.load_kaggle()
        import kaggle
        kaggle.api.authenticate()
        secrets = self.load_webdav()
        self.build_webdav_settings(secrets)
        if not skip_dependencies:
            self.load_dependencies()

    def __str__(self):
        return self.notebook
