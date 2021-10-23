from pathlib import Path

from fashionnets.util.io import all_paths_exist


def loader_info(name, variation):
    if "deep_fashion" in name:
        return deep_fashion_loader_info(variation)
    raise Exception("TODO")

def deep_fashion_loader_info(variation):
    return {
        "name": "deep_fashion_256",
        "variation": variation,#"df_quad_3",
        "preprocess": {
            "commands": [
                "mkdir -p ./deep_fashion_256",
                "mv ./train_256 ./deep_fashion_256",
                "mv ./validation_256 ./deep_fashion_256",
                "mv ./df_quad_3/train ./deep_fashion_256",
                "mv ./df_quad_3/validation ./deep_fashion_256",
                "rmdir ./df_quad_3"
            ],
            "check_existence": lambda: all_paths_exist(["./deep_fashion_256", "./deep_fashion"])
        }
    }