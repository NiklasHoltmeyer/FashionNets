





def list_jobs():
    input_shape = (144, 144)

    jobs = [
        {"back_bone_name": "resnet50", "weights": "mobile_net", "is_triplet": True},
        {"back_bone_name": "resnet50", "weights": None, "is_triplet": True},
        {"back_bone_name": "simplecnn", "weights": None, "is_triplet": True},
        {"back_bone_name": "resnet50", "weights": "mobile_net", "is_triplet": False},
        {"back_bone_name": "resnet50", "weights": None, "is_triplet": False},
        {"back_bone_name": "simplecnn", "weights": None, "is_triplet": False},
    ]

    return jobs, (144, 144)

