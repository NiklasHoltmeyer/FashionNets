from fashiondatasets.utils.logger.defaultLogger import defaultLogger

from fashionnets.callbacks.callbacks import callbacks
from fashionnets.models.SiameseModel import SiameseModel
from fashionnets.networks.SiameseNetwork import SiameseNetwork
from fashionnets.train_jobs.loader.checkpoint_loader import load_latest_checkpoint
from fashionnets.train_jobs.loader.job_loader import dump_settings


def sanity_check_job_settings(**train_job):
    optimizer = train_job["optimizer"]
    assert f"{optimizer.lr.numpy():.2e}" == train_job["learning_rate"]


def load_siamese_model_from_train_job(force_preprocess_layer=False, force_load_weights=False, **train_job):
    logger = defaultLogger("Load_Siamese_Model")
    logger.disabled = not train_job["verbose"]

    sanity_check_job_settings(**train_job)

    optimizer = train_job["optimizer"]

    back_bone_model = train_job["back_bone"]["embedding_model"]
    back_bone_preprocess_input_layer = train_job["back_bone"]["preprocess_input_layer"]

    if force_preprocess_layer:
        assert back_bone_preprocess_input_layer is not None

    siamese_network = SiameseNetwork(back_bone=back_bone_model,
                                     is_triplet=train_job["is_triplet"],
                                     input_shape=train_job["input_shape"],
                                     alpha=train_job["alpha"],
                                     beta=train_job["beta"],
                                     preprocess_input=back_bone_preprocess_input_layer,
                                     verbose=train_job["verbose"],
                                     channels=3)

    siamese_model = SiameseModel(siamese_network, back_bone_model)
    siamese_model.compile(optimizer=optimizer)

    siamese_model.fake_predict(train_job["input_shape"], train_job["is_triplet"])

    cp_path, run_name = train_job["path"]["checkpoint"], train_job["run"]["name"]

    logger.debug(f"Checkpoint: {cp_path}")
    logger.debug(f"Run Name: {run_name}")
    result_uploader = train_job["environment"].webdav

    _callbacks = callbacks(cp_path, run_name, keep_n=train_job.get("keep_n", 1),
                           verbose=train_job["verbose"], result_uploader=result_uploader)

    logger.debug(f"Callbacks: {_callbacks}")

    if train_job.get("load_weights", True):
        success, init_epoch = load_latest_checkpoint(siamese_model, **train_job)
        if success:
            print("Loaded Weights & Optimizer!")
        else:
            print("Loading Weights & Optimizer failed!")
            assert not force_load_weights, "force_load_weights is set, but no Checkpoints are found!"
    else:
        print("Load_Weights is set to False!")
        _checkpoint, init_epoch = None, 0

    freeze_layers = train_job.get("freeze_layers", None)

    if freeze_layers:
        siamese_model.siamese_network.back_bone = freeze_layers(siamese_model.siamese_network.back_bone)
    else:
        print("No layers frozen!")

    dump_settings(train_job)

    return siamese_model, init_epoch, _callbacks

# def load_backbone(checkpoint_path, input_shape, verbose, weights_path):
#    logger = defaultLogger("Load_Backbone")
#    logger.disabled = not verbose

#    run_name, back_bone, preprocess_input = load_backbone_info_resnet(input_shape, "resnet50", True, None)
#    siamese_network = SiameseNetwork(back_bone=back_bone,
#                                     is_triplet=True,
#                                     input_shape=input_shape,
#                                     alpha=1.0,
#                                     beta=0.5,
#                                     preprocess_input=preprocess_input,
#                                     verbose=verbose,
#                                     channels=3)

#    siamese_model = SiameseModel(siamese_network, back_bone)
#    init_epoch, _checkpoint = retrieve_checkpoint_info({
#        "path": {
#            "checkpoint": checkpoint_path
#        },
#        "run": {"name": run_name}
#    }, logger)
#    assert _checkpoint
#    logger.debug("Loading Weights!")
#    siamese_model.fake_predict(input_shape=input_shape, is_triplet=True)
#    siamese_model.load_weights(weights_path)
#    return siamese_model
##    return siamese_model
