from pathlib import Path




_dataset_paths = {}


def get_dataset_path(dataset_name):
    path = _dataset_paths.get(dataset_name, None)
    if path is None:
        raise ValueError(f"unknown dataset: {dataset_name}")

    return path
