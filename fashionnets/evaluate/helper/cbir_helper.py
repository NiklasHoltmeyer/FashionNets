from pathlib import Path

from fashiondatasets.deepfashion2.helper.pairs.similar_embeddings import find_top_k
from tqdm.auto import tqdm

from fashionnets.util.io import json_load, json_dump


def build_similar_idxs(run_name, epoch, embedding_json_path, use_tqdm=True):
    def retrieve_similar_idxs(_needles, _hay_stack):
        # Sadly cant be Optimized. Single Thread is faster then Multi-Threading. (Even if Data is not shared between
        # Threads)
        k = 101

        fn = lambda needle_: find_top_k([needle_], _hay_stack, similar=True, k=k)

        if use_tqdm:
            return list(map(fn, tqdm(_needles)))
        else:
            return list(map(fn, _needles))

    print(f"Build Similar Idxs {run_name} Epoch {epoch}")

    embedding_data = json_load(embedding_json_path)
    needles = embedding_data["query"]["embeddings"]
    hay_stack = embedding_data["gallery"]["embeddings"]

    similar_gallery_indexes = retrieve_similar_idxs(needles, hay_stack)

    root = Path(embedding_json_path).parent
    sim_idxs_items_path = Path(root, f"{epoch}_sim_idxs.json")
    json_dump(sim_idxs_items_path, similar_gallery_indexes)
