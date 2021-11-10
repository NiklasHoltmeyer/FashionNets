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
                "kaggle": "/gdrive/MyDrive/results/kaggle.json",
                "webdav": "/gdrive/MyDrive/results/webdav.json"
            },
            "checkpoint": "./results/",
            "dataset_base_path": "/content/",
            "tmp": "./tmp"
        }
    },
    "kaggle": {
        "secrets": {
            "kaggle": "kaggle",
            "webdav": "webdav"
        },
        "paths": {
            "checkpoint": "./results/",
            "dataset_base_path": "/kaggle/working/",
            "tmp": "./tmp"
        }
    },
    "local": {
        "paths": {
            "secrets": {
                "webdav": r"F:\workspace\webdav.json"
            },
            "checkpoint": r"F:\workspace\FashNets\\runs\\",
            "dataset_base_path": "F:\\workspace\\datasets\\",
            "tmp": "./tmp",
            "results_download_path": r"D:\masterarbeit_runs"
        }
    }
}
