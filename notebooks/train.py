notebook_name = "l_i2"
from fashionnets.train_jobs.loader.job_loader import load_job_settings, history_to_csv_string
from fashionnets.train_jobs.loader.model_loader import load_siamese_model_from_train_job

job_settings = load_job_settings(notebook_name, True)
siamese_model, init_epoch, callbacks = load_siamese_model_from_train_job(**job_settings)

datasets = job_settings["run"]["dataset"]
train_ds, val_ds = datasets["train"], datasets["val"]
history = siamese_model.fit(train_ds,
                            epochs=10,  #job_settings["epochs"]
                            validation_data=train_ds #val_ds
                            #callbacks=callbacks,
                            #initial_epoch=init_epoch
                            )

history_csv = history_to_csv_string(history, decimal_separator=",", **job_settings)
