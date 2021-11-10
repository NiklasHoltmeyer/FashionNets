from fashiondatasets.deepfashion2.helper.pairs.similar_embeddings import find_top_k
from fashiondatasets.utils.list import parallel_map

from fashionnets.util.io import json_load

embeddings_path = "D:\Download\embeddings.json"
embeddings = json_load(embeddings_path)


def retrieve_similar_idxs(needles, hay_stack, k=21):
    print("Jobs", len(needles))

    def job(needle):
        return find_top_k([needle], hay_stack, similar=True, k=k)

    return parallel_map(needles,
                        job,
                        desc="Retrieve Similar Items",
                        threads=8, total=len(needles))


similar_idxs = retrieve_similar_idxs(needles=embeddings["query"]["encodings"],
                                     hay_stack=embeddings["gallery"]["encodings"])
