#from fashionnets.dataloader.own import load_train_job

from fashionnets.train_jobs.jobs import load_job_from_notebook_name, load_siamese_model

train_job = load_job_from_notebook_name("l_h")

train_ds, val_ds = train_job["run"]["dataset"]["train"], train_job["run"]["dataset"]["val"]
result_uploader = train_job["environment"].webdav

siamese_model, init_epoch, callbacks = load_siamese_model(train_job, input_shape=train_job["input_shape"],
                                              verbose=train_job["verbose"],
                                              result_uploader=result_uploader)

