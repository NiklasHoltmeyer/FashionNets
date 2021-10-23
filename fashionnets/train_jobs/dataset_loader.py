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
            "operations": [
                ("rename", "./df_quad_3", "./deep_fashion"),
                ("mv", "./train_256/images", "./deep_fashion_256/train_256"),
                ("mv", "./validation_256/images", "./deep_fashion_256/validation_256"),
                ("rm", "./validation_256", None),
                ("rm", "./train_256", None),
            ],
            "check_existence": lambda: all_paths_exist(["./deep_fashion_256", "./deep_fashion"])
        }
    }