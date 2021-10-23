import json
import os
from pathlib import Path

from fashionnets.train_jobs.notebook_environments.Environment import Environment
from fashionnets.train_jobs.notebook_environments.consts import notebooks

try:
    # Imports that only work withing Kaggle
    from kaggle_secrets import UserSecretsClient
except:
    pass


class KaggleEnvironment(Environment):
    def __init__(self):
        super(KaggleEnvironment, self).__init__("kaggle")

    def load_webdav(self):
        assert self.train_job_name, "Set Run Name via KaggleEnv::set_name first"

        user_secrets = UserSecretsClient()
        secret_value_0 = user_secrets.get_secret(notebooks["kaggle"]["secrets"]["webdav"])
        return json.loads(secret_value_0)

    def load_kaggle(self):
        # Load Secrets
        if Path("~/.kaggle").exists():
            return

        os.system("mkdir ~/.kaggle")
        os.system("cp kaggle.json ~/.kaggle/")

        user_secrets = UserSecretsClient()
        secret_value_0 = user_secrets.get_secret(notebooks["kaggle"]["secrets"]["kaggle"])

        with open("/root/.kaggle/kaggle.json", "w+") as f:
            f.write(secret_value_0)
