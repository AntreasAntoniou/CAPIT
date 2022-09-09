"""
Storage associated utilities
"""
import os
import os.path
import pathlib
from typing import Dict, Union

import orjson as json


def save_json(filepath: Union[str, pathlib.Path], dict_to_store: Dict, overwrite=True):
    """
    Saves a metrics .json file with the metrics
    :param log_dir: Directory of log
    :param metrics_file_name: Name of .csv file
    :param dict_to_store: A dict of metrics to add in the file
    :param overwrite: If True overwrites any existing files with the same filepath,
    if False adds metrics to existing
    """

    if isinstance(filepath, pathlib.Path):
        filepath = str(filepath)

    metrics_file_path = filepath

    if overwrite and os.path.exists(metrics_file_path):
        pathlib.Path(metrics_file_path).unlink(missing_ok=True)

    parent_folder = "/".join(filepath.split("/")[:-1])
    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder, exist_ok=True)

    with open(metrics_file_path, "wb") as json_file:
        json_file.write(json.dumps(dict_to_store))


def load_json(filepath: Union[str, pathlib.Path]):
    """
    Loads the metrics in a dictionary.
    :param log_dir: The directory in which the log is saved
    :param metrics_file_name: The name of the metrics file
    :return: A dict with the metrics
    """

    if isinstance(filepath, pathlib.Path):
        filepath = str(filepath)

    with open(filepath, "rb") as json_file:
        dict_to_load = json.loads(json_file.read())

    return dict_to_load
