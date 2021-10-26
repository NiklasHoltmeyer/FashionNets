from fashionnets.train_jobs.loader.job_loader import prepare_environment, load_job_settings, history_to_csv_string
from fashionnets.train_jobs.loader.model_loader import load_siamese_model_from_train_job
import kaggle

job_worker_name = "l_i1"
# !pip install -q git+https://github.com/NiklasHoltmeyer/FashionNets.git

environment, training_job_cfg = prepare_environment(job_worker_name, debugging=False)

#kaggle.api.authenticate()
kaggle_downloader = kaggle.api.dataset_download_files  # <- doesnt work from withing the library. maybe a permissions issue

train_job = load_job_settings(environment=environment, training_job_cfg=training_job_cfg,
                                 kaggle_downloader=kaggle_downloader)

datasets = train_job["run"]["dataset"]
train_ds, val_ds = datasets["train"], datasets["val"]

result_uploader = train_job["environment"].webdav

train_ds = train_ds.take(1)
val_ds = val_ds.take(1)

siamese_model, init_epoch, callbacks = load_siamese_model_from_train_job(**train_job)

history = siamese_model.fit(train_ds,
                            epochs=1,  #job_settings["epochs"]
                            validation_data=train_ds #val_ds
                            #callbacks=callbacks,
                            #initial_epoch=init_epoch
                            )

history_csv = history_to_csv_string(history, decimal_separator=",", **train_job)
