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

from fashionnets.util.io import json_dump


class Evaluate:
    def __init__(self, dataset, model,
                 input_shape=(224, 224),
                 batch_size=32,
                 k=100):
        """
        Evaluate Backbone Top-K Accuracy
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

    def embed(self, key):
        images = Evaluate.paths_to_dataset(self.dataset[key]["paths"], self.input_shape, self.batch_size)

        embeddings = []

        for batch in tqdm(images, desc=f"Embed {key}"):
            embeddings.extend(self.model(batch))
        return embeddings

    def retrieve_most_similar(self):
        query_images = Evaluate.paths_to_dataset(self.dataset["query"]["paths"], self.input_shape, self.batch_size)

        most_similar_ids = []
        for batch in tqdm(query_images, desc="Retriev Most Similar"):
            query_embeddings = self.model.predict(batch)
            distances = distance_metric.cdist(query_embeddings, self.dataset["gallery"]["embeddings"], "cosine")

            for distance in distances:
                distance = 1 - distance
                idx_dist = list(zip(range(len(self.dataset["gallery"]["embeddings"])), distance))
                idx_dist = sorted(idx_dist, key=lambda d: d[1], reverse=True)[:self.k]
                most_sim_idxs = list(map(lambda d: d[0], idx_dist))
                most_sim_ids = list(map(lambda idx: self.dataset["gallery"]["ids"][idx], most_sim_idxs))
                most_similar_ids.append(most_sim_ids)
        return most_similar_ids

    def embed_dataset(self):
        for key in ["gallery", "query"]:
            if "embeddings" not in self.dataset[key].keys():
                self.dataset[key]["embeddings"] = self.embed(key)

    def dump_embeddings(self, path):
        """
            Dump Dataset. Calculate Embeddings Remote. Re-run Evaluations local
        """
        for key in ["gallery", "query"]:
            self.dataset[key]["embeddings"] = list(
                map(lambda tensor: tensor.numpy().tolist(), self.dataset[key]["embeddings"]))
            self.dataset[key].pop("paths")

        json_dump(path, self.dataset)

    def calculate(self):
        if "embeddings" not in self.dataset["gallery"].keys():
            self.dataset["gallery"]["embeddings"] = self.embed("gallery")
        assert len(self.dataset["gallery"]["paths"]) == len(self.dataset["gallery"]["embeddings"])

        most_similar = self.retrieve_most_similar()

        result = {}

        for k_ in [100, 50, 30, 25, 20, 15, 10, 5, 1]:
            if self.k < k_:
                continue
            hits = 0
            for gallery_retrieved, query_id in zip(most_similar, self.dataset["query"]["ids"]):
                if query_id in gallery_retrieved[:k_]:
                    hits += 1
            top_k = hits / len(self.dataset["query"]["ids"])
            result[str(k_)] = {
                "hits": hits,
                "total": len(self.dataset["query"]["ids"]),
                "accuracy": top_k
            }

        return result

    def validate_embeddings(self):
        assert (len(self.dataset["gallery"]["embeddings"])) == (len(self.dataset["gallery"]["ids"]))
        assert (len(self.dataset["query"]["embeddings"])) == (len(self.dataset["query"]["ids"]))



    @staticmethod
    def paths_to_dataset(paths, input_shape, batch_size):
        return paths.map(preprocess_image(input_shape)) \
            .batch(batch_size, drop_remainder=False) \
            .prefetch(tf.data.AUTOTUNE)
