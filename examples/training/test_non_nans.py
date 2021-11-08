notebook_name = "l_t_12_f100" #212
from fashionnets.train_jobs.loader.job_loader import load_job_settings, history_to_csv_string, prepare_environment
from fashionnets.train_jobs.loader.model_loader import load_siamese_model_from_train_job
from fashionnets.train_jobs.loader.job_loader import add_back_bone_to_train_job

environment, training_job_cfg = prepare_environment(notebook_name, debugging=True)

train_job = load_job_settings(environment=environment, training_job_cfg=training_job_cfg, kaggle_downloader=None)
train_job["batch_size"] = 1
job_settings = add_back_bone_to_train_job(environment=environment, **training_job_cfg)

siamese_model, init_epoch, _callbacks = load_siamese_model_from_train_job(**train_job,
                                                                          load_weights=True,
                                                                          force_preprocess_layer=True)