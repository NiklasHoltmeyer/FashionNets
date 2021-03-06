import tensorflow as tf
from fashiondatasets.own.helper.mappings import preprocess_image
from fashionnets.models.layer.Augmentation import compose_augmentations
from fashionnets.train_jobs.loader.dataset_loader import prepare_ds
from fashionnets.util.io import json_dump
from scipy.spatial import distance as distance_metric
from tensorflow import keras
from tqdm.auto import tqdm


class Evaluate:
    def __init__(self, dataset, model, job_settings,
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

        self.augmentation = compose_augmentations()(False)
        self.job_settings = job_settings

    def loss_validate(self, datasets, is_triplet, steps=None):
        test_ds_info, val_ds_info = datasets["test"], datasets["validation"]
        test_ds, val_ds = test_ds_info["dataset"], val_ds_info["dataset"]

        prepare_settings = {
            "input_shape": self.job_settings["input_shape"],
            "is_triplet": is_triplet,
            "batch_size": 32,
            "augmentation": compose_augmentations(),
            "generator_type": "apn"
        }

        test_ds, val_ds = prepare_ds(test_ds, is_train=False, **prepare_settings), prepare_ds(val_ds,
                                                                                              is_train=False,
                                                                                              **prepare_settings)
        test_val_ds = test_ds.concatenate(val_ds)
        test_eval = self.model.evaluate(test_ds, steps=steps)
        val_eval = self.model.evaluate(val_ds, steps=steps)
        test_val_eval = self.model.evaluate(test_val_ds, steps=steps)

        return {
            "test": test_eval, "validation": val_eval, "test+validation": test_val_eval
        }

        #    def embed(self, key):
        #        images = Evaluate.paths_to_dataset(self.dataset[key]["paths"], self.input_shape, self.batch_size)

        #        embeddings = []

        #        for batch in tqdm(images, desc=f"Embed {key}"):
        #            embeddings.extend(self.model(batch))
        #        return embeddings

        #    def retrieve_most_similar(self):
        #        query_images = Evaluate.paths_to_dataset(self.dataset["query"]["paths"], self.input_shape, self.batch_size)

        #        most_similar_ids = []
        #        for batch in tqdm(query_images, desc="Retrieve Most Similar"):
        #            query_embeddings = self.model.predict(batch)
        #            distances = distance_metric.cdist(query_embeddings, self.dataset["gallery"]["embeddings"], "cosine")

        #            for distance in distances:
        #                distance = 1 - distance
        #                idx_dist = list(zip(range(len(self.dataset["gallery"]["embeddings"])), distance))
        #                idx_dist = sorted(idx_dist, key=lambda d: d[1], reverse=True)[:self.k]
        #                most_sim_idxs = list(map(lambda d: d[0], idx_dist))
        #                most_sim_ids = list(map(lambda idx: self.dataset["gallery"]["ids"][idx], most_sim_idxs))
        #                most_similar_ids.append(most_sim_ids)
        #        return most_similar_ids

        #    def embed_dataset(self):
        #        for key in ["gallery", "query"]:
        #            if "embeddings" not in self.dataset[key].keys():
        #                self.dataset[key]["embeddings"] = self.embed(key)

        #    def dump_embeddings(self, path):
#        """
##            Dump Dataset. Calculate Embeddings Remote. Re-run Evaluations local
#        """
#        for key in ["gallery", "query"]:
#            if type(self.dataset[key]["embeddings"][0]) != list:
#                self.dataset[key]["embeddings"] = list(
# map(lambda tensor: tensor.numpy().tolist(), self.dataset[key]["embeddings"]))
#            if "paths" in self.dataset[key].keys():
#                self.dataset[key].pop("paths")

#        json_dump(path, self.dataset)

#    def calculate(self):
#        if "embeddings" not in self.dataset["gallery"].keys():
#            self.dataset["gallery"]["embeddings"] = self.embed("gallery")
#        assert len(self.dataset["gallery"]["paths"]) == len(self.dataset["gallery"]["embeddings"])

#        most_similar = self.retrieve_most_similar()

#        result = {}

#        for k_ in [100, 50, 30, 25, 20, 15, 10, 5, 1]:
#            if self.k < k_:
#                continue
#            hits = 0
#            for gallery_retrieved, query_id in zip(most_similar, self.dataset["query"]["ids"]):
#                if query_id in gallery_retrieved[:k_]:
# hits += 1
#            top_k = hits / len(self.dataset["query"]["ids"])
#            result[str(k_)] = {
#                "hits": hits,
#                "total": len(self.dataset["query"]["ids"]),
#                "accuracy": top_k
#            }

#        return result

#    def validate_embeddings(self):
#        assert (len(self.dataset["gallery"]["embeddings"])) == (len(self.dataset["gallery"]["ids"]))
#        assert (len(self.dataset["query"]["embeddings"])) == (len(self.dataset["query"]["ids"]))

#    @staticmethod
#    def paths_to_dataset(paths, input_shape, batch_size):
#        return paths.map(preprocess_image(input_shape)) \
#            .batch(batch_size, drop_remainder=False) \
#            .prefetch(tf.data.AUTOTUNE)
