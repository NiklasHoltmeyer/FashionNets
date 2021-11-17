import os

from fashionnets.train_jobs.loader.dataset_loader import build_dataset_hard_pairs_deep_fashion_1

notebook_name = "l_q_1e5aug_random_building"  # 212
#
from fashionnets.train_jobs.loader.job_loader import load_job_settings, history_to_csv_string, prepare_environment
from fashionnets.train_jobs.loader.model_loader import load_siamese_model_from_train_job
from fashionnets.train_jobs.loader.job_loader import add_back_bone_to_train_job
os.chdir(r'F:\workspace\FashNets\runs\1337_resnet50_imagenet_triplet')
environment, training_job_cfg = prepare_environment(notebook_name, debugging=True)

train_job = load_job_settings(environment=environment, training_job_cfg=training_job_cfg, kaggle_downloader=None)
job_settings = add_back_bone_to_train_job(environment=environment, **training_job_cfg)

datasets = train_job["run"]["dataset"]()

train_ds, val_ds = datasets["train"], datasets["val"]

# result_uploader = train_job["environment"].webdav
train_job["environment"].webdav = None

dataset = build_dataset_hard_pairs_deep_fashion_1(None,
                            job_settings,
                            init_epoch=1,
                            build_frequency=1)