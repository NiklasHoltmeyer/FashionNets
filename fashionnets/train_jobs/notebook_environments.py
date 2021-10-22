import json
import os
from pathlib import Path

from fashionnets.util.io import json_load, download_extract_kaggle
from fashionnets.util.remote import WebDav

try:
    # Imports that only work withing Kaggle
    from kaggle_secrets import UserSecretsClient
except:
    pass

try:
    # Imports that only work withing Google Colab
    from google.colab import drive
except:
    pass

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

    def set_name(self, name):
        self.train_job_name = name

    def load_webdav(self):
        pass

    def load_kaggle(self):
        pass

    def load_dependencies(self):
        if not self.download_dependencies:
            return
        kaggle_dependencies = self.dependencies.get("kaggle", [])
        [download_extract_kaggle(self.dataset_prefix + x, path="./", unzip=True) for x in kaggle_dependencies]

    def build_webdav_settings(self, webdav_secrets):
        if not webdav_secrets["base_path"].endswith("/"):
            webdav_secrets["base_path"] += "/"

        webdav_secrets["base_path"] += self.train_job_name

        self.webdav = WebDav(**webdav_secrets)

    def init(self):
        secrets = self.load_webdav()
        self.build_webdav_settings(secrets)

        self.load_kaggle()
        self.load_dependencies()

    def __str__(self):
        return self.notebook

class GoogleColabEnvironment(Environment):
    def __init__(self):
        super(GoogleColabEnvironment, self).__init__("google")

    def load_webdav(self):
        assert self.train_job_name, "Set Run Name via GoogleColabEnv::set_name first"
        return json_load("/gdrive/MyDrive/results/webdav.json")


    def load_kaggle(self):
        # Load Secrets
        if Path("~/.kaggle").exists():
            return

        drive.mount('/gdrive/')

        os.system("mkdir ~/.kaggle")

        if Path("/gdrive/MyDrive/results/kaggle.json").exists():
            json_path = "/gdrive/MyDrive/results/kaggle.json"
        else:
            from google.colab import files
            print("Please upload Kaggle.json")
            files.upload()
            json_path = "kaggle.json"
        os.system(f"cp {json_path} ~/.kaggle/")


class KaggleEnvironment(Environment):
    def __init__(self):
        super(KaggleEnvironment, self).__init__("kaggle")

    def load_webdav(self):
        assert self.train_job_name, "Set Run Name via KaggleEnv::set_name first"

        user_secrets = UserSecretsClient()
        secret_value_0 = user_secrets.get_secret("webdav")
        return json.loads(secret_value_0)

    def load_kaggle(self):
        # Load Secrets
        if Path("~/.kaggle").exists():
            return

        os.system("mkdir ~/.kaggle")
        os.system("cp kaggle.json ~/.kaggle/")

        user_secrets = UserSecretsClient()
        secret_value_0 = user_secrets.get_secret("kaggle")

        with open("/root/.kaggle/kaggle.json", "w+") as f:
            f.write(secret_value_0)


class LocalEnvironment(Environment):
    def __init__(self):
        super(LocalEnvironment, self).__init__("local", download_dependencies=False)

    def load_webdav(self):
        assert self.train_job_name, "Set Run Name via GoogleColabEnv::set_name first"
        return json_load(r"F:\workspace\webdav.json")


def environment(notebook_name):
    if notebook_name.startswith("g_"):
        return GoogleColabEnvironment()
    elif notebook_name.startswith("k_"):
        return KaggleEnvironment()
    elif notebook_name.startswith("l_"):
        return LocalEnvironment()
    else:
        raise Exception(f"Unsupported Notebook Prefix {notebook_name}")
