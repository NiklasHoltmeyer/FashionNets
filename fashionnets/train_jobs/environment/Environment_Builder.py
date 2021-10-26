from fashionnets.train_jobs.environment.Environment_Consts import job_worker_name_prefixes
from fashionnets.train_jobs.environment.local.LocalEnvironment import LocalEnvironment
from fashionnets.train_jobs.environment.remote.GoogleColabEnvironment import GoogleColabEnvironment
from fashionnets.train_jobs.environment.remote.KaggleEnvironment import KaggleEnvironment


class EnvironmentBuilder:
    @staticmethod
    def by_runtime_notebook_name(notebook_name):
        if notebook_name.startswith(job_worker_name_prefixes["google"]):
            return GoogleColabEnvironment()
        elif notebook_name.startswith(job_worker_name_prefixes["kaggle"]):
            return KaggleEnvironment()
        elif notebook_name.startswith(job_worker_name_prefixes["local"]):
            return LocalEnvironment()
        else:
            raise Exception(f"Unsupported Notebook Prefix {notebook_name}")
