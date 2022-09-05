"""
Storage associated utilities
"""
import logging as log
import os
import os.path
import pathlib
from dataclasses import dataclass
from typing import Dict

import orjson as json
import requests
import tqdm  # progress bar
import yaml
from google.cloud import storage
from omegaconf import DictConfig

from tali.utils.arg_parsing import DictWithDotNotation


def build_experiment_folder(experiment_name, log_path, save_images=True):
    """
    An experiment logging folder goes along with each  This builds that
    folder
    :param args: dictionary of arguments
    :return: filepaths for saved models, logs, and images
    """
    saved_models_filepath = os.path.join(
        log_path, experiment_name.replace("%.%", "/"), "saved_models"
    )
    logs_filepath = os.path.join(
        log_path, experiment_name.replace("%.%", "/"), "summary_logs"
    )
    images_filepath = os.path.join(
        log_path, experiment_name.replace("%.%", "/"), "images"
    )

    if not os.path.exists(logs_filepath):
        os.makedirs(logs_filepath)

    if not os.path.exists(saved_models_filepath):
        os.makedirs(saved_models_filepath)

    if not os.path.exists(images_filepath):
        os.makedirs(images_filepath)

    if save_images:
        if not os.path.exists(images_filepath + "/train"):
            os.makedirs(images_filepath + "/train")
        if not os.path.exists(images_filepath + "/val"):
            os.makedirs(images_filepath + "/val")
        if not os.path.exists(images_filepath + "/test"):
            os.makedirs(images_filepath + "/test")

    return saved_models_filepath, logs_filepath, images_filepath


def download_file_from_url(url, filename=None, verbose=False):
    """
    Download file with progressbar
    __author__ = "github.com/ruxi"
    __license__ = "MIT"
    Usage:
        download_file('http://web4host.net/5MB.zip')
    """
    if not filename:
        local_filename = os.path.join("", url.split("/")[-1])
    else:
        local_filename = filename
    r = requests.get(url, stream=True)
    file_size = int(r.headers["Content-Length"])
    chunk_size = 1024
    num_bars = file_size // chunk_size
    if verbose:
        log.info(dict(file_size=file_size))
        log.info(dict(num_bars=num_bars))

    file_directory = filename.replace(filename.split("/")[-1], "")

    file_directory = pathlib.Path(file_directory)

    file_directory.mkdir(parents=True, exist_ok=True)

    with open(local_filename, "wb") as fp:
        for chunk in tqdm.tqdm(
            r.iter_content(chunk_size=chunk_size),
            total=num_bars,
            unit="KB",
            desc=local_filename,
            leave=True,  # progressbar stays
        ):
            fp.write(chunk)
    return local_filename


def save_json(filepath, metrics_dict, overwrite=True):
    """
    Saves a metrics .json file with the metrics
    :param log_dir: Directory of log
    :param metrics_file_name: Name of .csv file
    :param metrics_dict: A dict of metrics to add in the file
    :param overwrite: If True overwrites any existing files with the same filepath,
    if False adds metrics to existing
    """

    if isinstance(filepath, pathlib.Path):
        filepath = str(filepath)

    if ".json" not in filepath:
        filepath = f"{filepath}.json"

    metrics_file_path = filepath

    if overwrite and os.path.exists(metrics_file_path):
        pathlib.Path(metrics_file_path).unlink(missing_ok=True)

    parent_folder = "/".join(filepath.split("/")[:-1])
    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder, exist_ok=True)

    # 'r'       open for reading (default)
    # 'w'       open for writing, truncating the file first
    # 'x'       create a new file and open it for writing
    # 'a'       open for writing, appending to the end of the file if it exists
    # 'b'       binary mode
    # 't'       text mode (default)
    # '+'       open a disk file for updating (reading and writing)
    # 'U'       universal newline mode (deprecated)
    # log.info(f"{isinstance(metrics_dict, dict)} {len(metrics_dict)}")
    # folder_keys = list(metrics_dict.keys())[:5]
    #
    # for folder_key in folder_keys:
    #     log.info(f"{metrics_dict[folder_key]}")

    with open(metrics_file_path, "wb") as json_file:
        json_file.write(json.dumps(metrics_dict))


def load_json(filepath):
    """
    Loads the metrics in a dictionary.
    :param log_dir: The directory in which the log is saved
    :param metrics_file_name: The name of the metrics file
    :return: A dict with the metrics
    """

    if isinstance(filepath, pathlib.Path):
        filepath = str(filepath)

    if ".json" not in filepath:
        filepath = f"{filepath}.json"

    with open(filepath, "rb") as json_file:
        metrics_dict = json.loads(json_file.read())

    return metrics_dict


def save_model_config_to_yaml(model_name, config_attributes, config_filepath):
    config_filepath = pathlib.Path(config_filepath)
    try:
        with open(config_filepath, mode="w+") as file_reader:
            config = yaml.safe_load(file_reader)
            config[model_name] = config_attributes
            yaml.dump(config, file_reader)
            return True
    except Exception:
        log.exception("Could not save model config to yaml")
        return False


def load_model_config_from_yaml(config_filepath, model_name):
    config_filepath = pathlib.Path(config_filepath)
    with open(config_filepath, mode="r") as file_reader:
        config = yaml.safe_load(file_reader)

    return DictWithDotNotation(config[model_name])


from yaml import CLoader as Loader, CDumper as Dumper

# dump = yaml.dump(
#     dummy_data, fh, encoding="utf-8", default_flow_style=False, Dumper=Dumper
# )
# data = yaml.load(fh, Loader=Loader)


def save_yaml(filepath, object_to_store):
    try:
        with open(filepath, mode="w+") as file_reader:
            yaml.dump(
                object_to_store,
                file_reader,
                encoding="utf-8",
                default_flow_style=False,
                Dumper=Dumper,
            )
            return True
    except Exception:
        return False


def load_yaml(filepath):
    """
    Loads the metrics in a dictionary.
    :param log_dir: The directory in which the log is saved
    :param metrics_file_name: The name of the metrics file
    :return: A dict with the metrics
    """

    if isinstance(filepath, pathlib.Path):
        filepath = os.fspath(filepath.resolve())

    if ".yaml" not in filepath:
        filepath = f"{filepath}.yaml"

    with open(filepath) as yaml_file:
        config_dict = yaml.safe_load(yaml_file, Loader=Loader)
    # log.debug(f"Loaded yaml file: {filepath}")
    return config_dict


def pretty_print_dict(input_dict, tab=0):
    output_string = []
    current_tab = tab
    tab_string = "".join(current_tab * ["\t"])
    for key, value in input_dict.items():

        # logging.info(f'{key} is {type(value)}')

        if isinstance(value, (Dict, DictConfig)):
            # logging.info(f'{key} is Dict')
            value = pretty_print_dict(value, tab=current_tab + 1)

        output_string.append(f"\n{tab_string}{key}: {value}")
    return "".join(output_string)


def create_bucket_class_location(bucket_name, storage_class, location):
    """
    Create a new bucket in the US region with the coldline storage
    class
    """
    # bucket_name = "your-new-bucket-name"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    bucket.storage_class = storage_class
    new_bucket = storage_client.create_bucket(bucket, location=location)

    log.info(
        f"Created bucket {new_bucket.name} "
        f"in {new_bucket.location} "
        f"with storage class {new_bucket.storage_class}"
    )
    return new_bucket


def get_bucket_client(bucket_name):
    storage_client = storage.Client()
    bucket_client = storage_client.bucket(bucket_name)

    return storage_client, bucket_client


def upload_file(
    bucket_name, source_file_name, destination_file_name, bucket_client=None
):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    if bucket_client is None:
        storage_client = storage.Client()
        bucket_client = storage_client.bucket(bucket_name)

    blob = bucket_client.blob(destination_file_name)

    blob.upload_from_filename(source_file_name)

    log.info(f"File {source_file_name} uploaded to {destination_file_name}")


def download_file(
    bucket_name, source_file_name, destination_file_name, bucket_client=None
):
    """Downloads a blob from the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The ID of your GCS object
    # source_blob_name = "storage-object-name"

    # The path to which the file should be downloaded
    # destination_file_name = "local/path/to/file"

    if bucket_client is None:
        storage_client = storage.Client()
        bucket_client = storage_client.bucket(bucket_name)
    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket_client.blob(source_file_name)
    blob.download_to_filename(destination_file_name)

    log.info(
        f"Downloaded storage object {source_file_name} "
        f"from bucket {bucket_name} "
        f"to local file {destination_file_name}"
    )


@dataclass
class GoogleStorageClientConfig:
    local_log_dir: str
    bucket_name: str
    experiment_name: str


# Use cases: 1. (New) Brand new experiment: It should scan the directory structure as
# it is, delete whatever is on google storage and upload the new structure and keep
# updating it periodically. - rsync with 'syncing local to remote' from local,
# then in subsequent updates just do default rsync 2. (Continued-full or
# Continued-minimal) Continued experiment: It should find the online experiment
# files, compare with local files and download the missing files OR just download
# necessary files to continue - Continued full, full rsync from bucket to local,
# continued in all updates - Continued minimal, full rsync from bucket to local
# excluding a prefix or postfix that ensures all weights other than 'latest' are not
# synced

#


def google_storage_rsync_gs_to_local(
    bucket_name,
    experiments_root_dir,
    experiment_name,
    exclude_list,
    options_list,
):

    options_string = "".join(options_list) if len(options_list) > 0 else ""
    exclude_string = (
        " ".join([f"-x {exclude_item}" for exclude_item in exclude_list])
        if len(exclude_list) > 0
        else ""
    )
    command_string = f"gsutil -m rsync -{options_string} {exclude_string} gs://{bucket_name}/{experiment_name}/ {experiments_root_dir}/;"

    screen_command_string = (
        f"screen -dmS gsutil-update bash -c '{command_string}; exec bash'"
    )
    log.debug(command_string + "\n\n")
    os.system(screen_command_string)


def google_storage_rsync_local_to_gs(
    bucket_name,
    experiments_root_dir,
    experiment_name,
    exclude_list,
    options_list,
):
    options_string = "".join(options_list) if len(options_list) > 0 else ""
    exclude_string = (
        " ".join([f"-x {exclude_item}" for exclude_item in exclude_list])
        if len(exclude_list) > 0
        else ""
    )
    command_string = (
        f"gsutil -m rsync -{options_string} {exclude_string} {experiments_root_dir}/ gs://{bucket_name}/{experiment_name}/;"
        # f"wandb artifact cache cleanup 1GB"
    )

    screen_command_string = (
        f"screen -dmS gsutil-update bash -c '{command_string}; exec bash'"
    )

    log.debug(command_string + "\n\n")
    os.system(screen_command_string)


class GoogleStorageClient(object):
    def __init__(self, config: GoogleStorageClientConfig):
        super(GoogleStorageClient, self).__init__()
        self.config = config
        self.directory_structure_cache = None
        self.bucket_name = config.bucket_name
        self.storage_client, self.bucket_client = get_bucket_client(
            bucket_name=self.bucket_name
        )

    def get_remote_dir_tree(self):
        self.bucket_client.list_blobs(
            max_results=None,
            page_token=None,
            prefix=self.config.experiment_name,
            delimiter="/",
            start_offset=None,
            end_offset=None,
            include_trailing_delimiter=None,
            versions=None,
            projection="noAcl",
            fields=None,
            client=None,
        )

    def upload_file(self, filepath_on_bucket, local_filepath):
        upload_file(
            bucket_name=self.bucket_name,
            source_file_name=local_filepath,
            destination_file_name=filepath_on_bucket,
        )

    def download_file(self, filepath_on_bucket, local_filepath):
        download_file(
            bucket_name=self.bucket_name,
            source_file_name=filepath_on_bucket,
            destination_file_name=local_filepath,
        )
