from fashionnets.train_jobs.notebook_environments.Environment import Environment
from fashionnets.train_jobs.notebook_environments.consts import notebooks
from fashionnets.util.io import json_load

class LocalEnvironment(Environment):
    def __init__(self):
        super(LocalEnvironment, self).__init__("local", download_dependencies=False)

    def load_webdav(self):
        assert self.train_job_name, "Set Run Name via GoogleColabEnv::set_name first"
        return json_load(notebooks["local"]["paths"]["secrets"]["webdav"])