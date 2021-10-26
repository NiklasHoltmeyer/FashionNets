from fashionnets.train_jobs.environment.Environment import Environment
from fashionnets.train_jobs.environment.Environment_Consts import notebooks
from fashionnets.util.io import json_load


class LocalEnvironment(Environment):
    def __init__(self):
        super(LocalEnvironment, self).__init__("local", download_dependencies=False)

    def load_webdav(self):
        assert self.train_job_name, "Set Run Name via GoogleColabEnv::set_name first"
        return json_load(notebooks["local"]["paths"]["secrets"]["webdav"])
