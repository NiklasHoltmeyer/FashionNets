from tensorflow import keras


class SaveEmbeddingModel(keras.callbacks.Callback):
    """
    Its just like CSV-Logger. But with Custom Seperator und Save Strategy.
    Some how the normal CSV-Logger does not Append on Google Colab.
    """

    def __init__(self, model_cp_path):
        super(SaveEmbeddingModel, self).__init__()
        self.model_cp_path = model_cp_path

    def on_epoch_end(self, epoch, logs=None):
        self.model.save_backbone(self.model_cp_path, epoch)
