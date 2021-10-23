job_worker_name_prefixes = {
    "google": "g_",
    "kaggle": "k_",
    "local": "l_"
}

kaggle = {
    "datasets": {
        "prefix": "masterokay/"
    },
    "paths": {
        "secret": "~/.kaggle/kaggle.json"
    }
}

notebooks = {
    "google": {
        "paths": {
            "secrets": {
                "kaggle": "/gdrive/MyDrive/results/kaggle.json"
            },
            "checkpoint": "./", #"/gdrive/MyDrive/results/",
            "dataset_base_path": "/content/"
        }
    },
    "kaggle": {
        "secrets": {
            "kaggle": "kaggle",
            "webdav": "webdav"
        },
        "paths":{
            "checkpoint": "./",
            "dataset_base_path": "/kaggle/working/"
        }
    },
    "local": {
        "paths": {
            "secrets": {
                "webdav": r"F:\workspace\webdav.json"
            },
            "checkpoint": r"F:\workspace\FashNets\\",
            "dataset_base_path": "F:\\workspace\\datasets\\"
        }
    }
}
