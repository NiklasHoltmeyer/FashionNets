from fashionnets.train_jobs.notebook_environments.GoogleColabEnvironment import GoogleColabEnvironment
from fashionnets.train_jobs.notebook_environments.KaggleEnvironment import KaggleEnvironment
from fashionnets.train_jobs.notebook_environments.LocalEnvironment import LocalEnvironment
from fashionnets.train_jobs.notebook_environments.consts import job_worker_name_prefixes


def env_by_name(notebook_name):
    if notebook_name.startswith(job_worker_name_prefixes["google"]):
        return GoogleColabEnvironment()
    elif notebook_name.startswith(job_worker_name_prefixes["kaggle"]):
        return KaggleEnvironment()
    elif notebook_name.startswith(job_worker_name_prefixes["local"]):
        return LocalEnvironment()
    else:
        raise Exception(f"Unsupported Notebook Prefix {notebook_name}")
