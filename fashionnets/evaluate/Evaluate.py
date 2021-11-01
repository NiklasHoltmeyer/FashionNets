from fashionnets.train_jobs.loader.model_loader import load_siamese_model_from_train_job, load_backbone
from fashionnets.train_jobs.loader.backbone_loader import load_backbone_info_resnet
from collections import defaultdict
import tensorflow as tf
from fashiondatasets.deepfashion2.DeepFashionCBIR import DeepFashionCBIR
from fashiondatasets.own.helper.mappings import preprocess_image
from fashiondatasets.own.helper.mappings import preprocess_image
import numpy as np
from scipy.spatial import distance as distance_metric

from tqdm.auto import tqdm
from tensorflow import keras


class Evaluate:
    def __init__(self, dataset, model,
                 input_shape=(224, 224),
                 batch_size=32,
                 k=20):
        """

        :param dataset: fashiondatasets.deepfashion2.DeepFashionCBIR like Dataset
        :param model: Tensorflow Model or String of Checkpoint
        :param input_shape: Input Shape of Model. ResNet-50 Default is (224, 224)
        :param k:
        """
        self.dataset = dataset
        self.input_shape = input_shape
        self.k = k
        self.batch_size = batch_size

        if type(model) == str:
            self.model = keras.models.load_model(model)
        else:
            self.model = model

    def encode_gallery(self):
        gallery_images = Evaluate.paths_to_dataset(self.dataset["gallery"]["paths"], self.input_shape, self.batch_size)

        encodings = []
        for batch in tqdm(gallery_images, desc="Encode Gallery"):
            encodings.extend(self.model(batch))

        return encodings

    def retrieve_most_similar(self):
        query_images = Evaluate.paths_to_dataset(self.dataset["query"]["paths"], self.input_shape, self.batch_size)

        most_similar_ids = []
        for batch in tqdm(query_images, desc="Retriev Most Similar"):
            query_embeddings = self.model.predict(batch)
            distances = distance_metric.cdist(query_embeddings, self.dataset["gallery"]["encodings"], "cosine")

            for distance in distances:
                distance = 1 - distance
                idx_dist = list(zip(range(len(self.dataset["gallery"]["encodings"])), distance))
                idx_dist = sorted(idx_dist, key=lambda d: d[1], reverse=True)[:self.k]
                most_sim_idxs = list(map(lambda d: d[0], idx_dist))
                most_sim_ids = list(map(lambda idx: self.dataset["gallery"]["ids"][idx], most_sim_idxs))
                most_similar_ids.append(most_sim_ids)
        return most_similar_ids

    def calculate(self):
        self.dataset["gallery"]["encodings"] = self.encode_gallery()
        assert len(self.dataset["gallery"]["paths"]) == len(self.dataset["gallery"]["encodings"])

        most_similar = self.retrieve_most_similar()

        hits = 0
        for gallery_retrieved, query_id in zip(most_similar, self.dataset["query"]["ids"]):
            if query_id in gallery_retrieved:
                hits += 1
        top_k = hits / len(self.dataset["query"]["ids"])

        return {
            "top_k": top_k
        }

    @staticmethod
    def paths_to_dataset(paths, input_shape, batch_size):
        return paths.map(preprocess_image(input_shape)) \
            .batch(batch_size, drop_remainder=False) \
            .prefetch(tf.data.AUTOTUNE)
