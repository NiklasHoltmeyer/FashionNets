from fashiondatasets.deepfashion2.helper.pairs.similar_embeddings import find_top_k
from tqdm.auto import tqdm
from fashiondatasets.utils.list import parallel_map

from fashionnets.util.io import json_load
from tqdm.auto import tqdm
embeddings_path = "D:\Download\embeddings.json"
embeddings = json_load(embeddings_path)


def retriev_similar_idxs(needles, hay_stack, k=21):
    print("Jobs", len(needles))

    def job(needle):
        return find_top_k([needle], hay_stack, reverse=True, k=k)

    return parallel_map(needles,
                        job,
                        desc="Retriev Similar Items",
                        threads=8, total=len(needles))

similar_idxs = retriev_similar_idxs(needles=embeddings["query"]["encodings"], hay_stack=embeddings["gallery"]["encodings"])