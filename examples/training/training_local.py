import logging
import os

from fashiondatasets.utils.logger.defaultLogger import defaultLogger

from fashionnets.models.embedding.resnet50 import ResNet50Builder

notebook_name = "l_q_fine_freeze_3"  # 212 t_test_ctl
#
from fashionnets.train_jobs.loader.job_loader import load_job_settings, prepare_environment
from fashionnets.train_jobs.loader.model_loader import load_siamese_model_from_train_job
from fashionnets.train_jobs.loader.job_loader import add_back_bone_to_train_job
from tqdm.auto import tqdm
defaultLogger().setLevel(logging.DEBUG)
# t_apn
# q_apn
os.chdir(r'F:\workspace\FashNets\runs\1337_resnet50_imagenet_triplet')
environment, training_job_cfg = prepare_environment(notebook_name, debugging=True)
training_job_cfg["nrows"] = None

train_job = load_job_settings(environment=environment, training_job_cfg=training_job_cfg, kaggle_downloader=None)
job_settings = add_back_bone_to_train_job(environment=environment, **training_job_cfg)

siamese_model, init_epoch, _callbacks = load_siamese_model_from_train_job(**train_job,
                                                                          load_weights=False,
                                                                          force_preprocess_layer=True)
siamese_model.siamese_network.back_bone = ResNet50Builder.freeze_first_n_layers(siamese_model.siamese_network.back_bone__, 30)

x = sum([l.trainable for l in siamese_model.siamese_network.back_bone__.layers])
print(x, len(siamese_model.siamese_network.back_bone__.layers) - x)
print("WUHU")
exit(0)
datasets = train_job["run"]["dataset"]() # für ctl auf own nach load model

train_ds, val_ds = datasets["train"], datasets["val"]

# result_uploader = train_job["environment"].webdav
train_job["environment"].webdav = None

# train_ds = train_ds.take(1)0

history = siamese_model.fit(train_ds,
                            epochs=20,  # 2,  # job_settings["epochs"]
                            validation_data=val_ds,  # val_ds
                            # callbacks=_callbacks,
                            validation_steps=1,
                            # steps_per_epoch=1,
                            # initial_epoch=init_epoch
                            )

# history_csv = history_to_csv_string(history, decimal_separator=",", **job_settings)
