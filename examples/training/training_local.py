notebook_name = "l_b1_11"
from fashionnets.train_jobs.loader.job_loader import load_job_settings, history_to_csv_string, prepare_environment
from fashionnets.train_jobs.loader.model_loader import load_siamese_model_from_train_job
from fashionnets.train_jobs.loader.job_loader import add_back_bone_to_train_job

environment, training_job_cfg = prepare_environment(notebook_name, debugging=True)

train_job = load_job_settings(environment=environment, training_job_cfg=training_job_cfg, kaggle_downloader=None)
job_settings = add_back_bone_to_train_job(environment=environment, **training_job_cfg)

datasets = train_job["run"]["dataset"]()
train_ds, val_ds = datasets["train"], datasets["val"]
#result_uploader = train_job["environment"].webdav
train_job["environment"].webdav = None

siamese_model, init_epoch, _callbacks = load_siamese_model_from_train_job(**train_job,
                                                                          load_weights=True,
                                                                          force_preprocess_layer=True)
history = siamese_model.fit(train_ds,
                            epochs=1,#2,  # job_settings["epochs"]
                            validation_data=train_ds,  # val_ds
                            #callbacks=_callbacks,
                            validation_steps=1,
                            steps_per_epoch=1,
                            # initial_epoch=init_epoch
                            )

history_csv = history_to_csv_string(history, decimal_separator=",", **job_settings)
