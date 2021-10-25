job_worker_name = "l_i2"

from fashionnets.train_jobs.loader.model_loader import load_siamese_model_from_train_job
from fashionnets.train_jobs.jobs import load_train_job, load_job_f_settings, load_job_info_from_notebook_name

settings = load_job_info_from_notebook_name(job_worker_name)
##
#@title Prepare Dataset
train_job = load_job_f_settings(**settings)
#train_job = {**additional_settings, **settings, **train_job}
datasets = train_job["run"]["dataset"]
train_ds, val_ds = datasets["train"], datasets["val"]

result_uploader = settings["environment"].webdav
##
siamese_model, init_epoch, _callbacks = load_siamese_model_from_train_job(**train_job, load_weights=False)
history_1 = siamese_model.fit(train_ds, epochs=10)
