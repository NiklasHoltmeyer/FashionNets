import os

from fashionnets.train_jobs.notebook_environments.consts import kaggle
from fashionnets.util.remote import WebDav

"""Helper Classes to Init Training-Environment (Setting up Secrets, Defining Env Based Settings ...)
"""


class Environment:
    def __init__(self, notebook, download_dependencies=True, dataset_prefix="masterokay/"):
        self.notebook = notebook
        self.train_job_name = None
        self.webdav = None
        self.dependencies = {}
        self.download_dependencies = download_dependencies
        self.dataset_prefix = dataset_prefix

        print("Environment:", self)

    def set_name(self, name):
        self.train_job_name = name

    def load_webdav(self):
        pass

    def load_kaggle(self):
        pass

    def prepare(self, skip_preprocess):
        preprocess_settings = self.dependencies["kaggle"].get("preprocess", None)

        if not preprocess_settings:
            print("Preprocess not set")
            return

        if skip_preprocess:
            print("Skip Preprocess set")
            return

        for cmd in self.dependencies["kaggle"]["preprocess"]["commands"]:
            try:
                os.system(cmd)
            except:
                print("Exception", cmd)
                pass

    def load_dependencies(self, kaggle_downloader=None, skip_preprocess=False):
        if not self.download_dependencies:
            return

        paths_exist = self.dependencies["kaggle"].get("preprocess", {}).get("check_existence", lambda: False)

        if paths_exist():
            print("All Paths already exist!")
            return

        if kaggle_downloader:
            for ds_name in self.dependencies["kaggle"].values():
                if type(ds_name) is not str:
                    continue
                ds_full_name = kaggle["datasets"]["prefix"] + ds_name.replace("_", "-")
                print("Download:", ds_full_name)
                kaggle_downloader(ds_full_name, f"./", unzip=True)

            self.prepare(skip_preprocess)

            # preprocess
            # operations

    def build_webdav_settings(self, webdav_secrets):
        if not webdav_secrets["base_path"].endswith("/"):
            webdav_secrets["base_path"] += "/"

        webdav_secrets["base_path"] += self.train_job_name

        self.webdav = WebDav(**webdav_secrets)

    def init(self):
        self.load_kaggle()
        import kaggle
        kaggle.api.authenticate()
        secrets = self.load_webdav()
        self.build_webdav_settings(secrets)

        self.load_dependencies()

    def __str__(self):
        return self.notebook
