from tensorflow import keras


def base_settings(debugging, learning_rate=1e-4):
    optimizer = keras.optimizers.Adam(learning_rate)
    base = {
        "input_shape": (224, 224),  # (144, 144),
        "alpha": 1.0,
        "beta": 0.5,
        "epochs": 20,
        "verbose": False,
        "nrows": None,
        "buffer_size": 32,
        "batch_size": 32,
        "optimizer": optimizer,
        "learning_rate": f"{optimizer.lr.numpy():.2e}"
    }

    base["target_shape"] = base["input_shape"]

    debug_cfg = {
        "nrows": 20,
        "verbose": True,
        "batch_size": 1,
        "epochs": 20
    }

    if debugging:
        return {**base, **debug_cfg}

    return base


def back_bone_settings(back_bone_name, weights, is_triplet):
    assert back_bone_name in ["resnet50", "simplecnn"]
    assert weights in ["imagenet", None]
    assert is_triplet in [True, False]

    return {
        "back_bone_name": back_bone_name, "weights": weights, "is_triplet": is_triplet
    }
