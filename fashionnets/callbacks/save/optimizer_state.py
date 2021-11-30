from pathlib import Path

from fashiondatasets.utils.logger.defaultLogger import defaultLogger
from tensorflow import keras

from fashionnets.models.states.OptimizerState import OptimizerState

logger = defaultLogger("deepfashion_callbacks")

# noinspection PyUnresolvedReferences
class SaveOptimizerState(keras.callbacks.Callback):
    def __init__(self, checkpoint_path, name):
        super(SaveOptimizerState, self).__init__()

        self.model_cp_latest_path = Path(checkpoint_path, name + "_opt-")
        self.model_cp_latest_path.parent.mkdir(parents=True, exist_ok=True)

        self.model_cp_latest_path = str(self.model_cp_latest_path.resolve())

    def on_epoch_end(self, epoch, logs=None):
        try:
            cp_path = self.model_cp_latest_path + f"{epoch:04d}.pkl"
            path = str(Path(cp_path).resolve())

            OptimizerState(self.model.optimizer).save(path)

        except Exception as e:
            logger.error("CustomSaveOptimizerState::on_epoch_end")  # easier to trace Exception from withing Google Colab
            raise Exception(e)


