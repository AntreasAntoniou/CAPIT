#!/usr/bin/env python3

"""convert a range of different audio files to mp3 files
using multiple parallel processes"""

import argparse
import concurrent.futures
import logging
import multiprocessing as mp
import os
import pathlib
import pprint
import shutil
import sys
from collections import defaultdict

import numpy as np
import tqdm
from rich.logging import RichHandler

from gate.utils.logging_helpers import get_logging
from preprocessing_scripts.convert_audiofiles_to_npz import path_to_string

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
ch = RichHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter("%(levelname)s - %(message)s")

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
log.addHandler(ch)


def get_base_arguments():
    parser = argparse.ArgumentParser()
    # data and I/O
    parser.add_argument("--dataset_filepath", type=str, default="source_data/")
    parser.add_argument("--logger_level", type=str, default="INFO")
    parser.add_argument("--num_processes", type=int, default=mp.cpu_count())
    parser.add_argument("--dataset_size_identifier", type=str, default="deci")

    return parser.parse_args()


def delete_file_if_exists(path: pathlib.Path):
    logging.debug(f"Deleting {path}")
    if path.exists():
        path.unlink()


def process_files(file_tuples):
    dataset_dir, json_file_path, training_set_size_fraction_value = file_tuples
    json_file_path = pathlib.Path(f"{dataset_dir}/{json_file_path}")
    video_files = list(json_file_path.parent.glob("**/*.frames"))
    files_to_be_deleted = np.random.choice(
        video_files,
        size=int(len(video_files) * (1.0 - training_set_size_fraction_value)),
        replace=False,
    )
    total = len(video_files)
    deleted = len(files_to_be_deleted)
    try:
        for file in files_to_be_deleted:
            log.debug(f"Deleting {file}")
            shutil.rmtree(file)
            delete_file_if_exists(
                pathlib.Path(path_to_string(file).replace(".frames", ".npz"))
            )

        return True, total, deleted, file_tuples
    except Exception as e:
        log.debug(f"Exception: {e}")
        return False, total, deleted, file_tuples


if __name__ == "__main__":
    args = get_base_arguments()
    logging = get_logging(args.logger_level)

    num_youtube_video_folder_dict = {"train": 141468, "val": 6369, "test": 6500}
    percentage_to_keep = {
        "milli": 0.001,
        "centi": 0.01,
        "deci": 0.1,
        "base": 1.0,
    }[args.dataset_size_identifier]
    num_youtube_video_folders = np.sum(
        value for value in num_youtube_video_folder_dict.values()
    )

    if not os.path.exists(args.dataset_filepath):
        logging.debug(f"Source path {args.dataset_filepath} not found")

    failed_jobs = []
    matching_files = defaultdict(list)
    # get all of the source audio filenames
    logging.info(f"Current working directory is {args.dataset_filepath}")

    matched_meta_data_files = []
    with tqdm.tqdm(total=num_youtube_video_folders, smoothing=0.0) as pbar:
        for dir_path in pathlib.Path(args.dataset_filepath).iterdir():
            cur_file = dir_path / "meta_data.json"
            if cur_file.exists():
                meta_data_string_path = path_to_string(cur_file).replace(
                    args.dataset_filepath, ""
                )
                matched_meta_data_files.append(meta_data_string_path)
            pbar.update(1)

    logging.info(f"Found {len(matched_meta_data_files)} matched meta_data files")

    file_tuples = [
        (args.dataset_filepath, item, percentage_to_keep)
        for item in matched_meta_data_files
    ]

    logging.info("Scanning folders for media files")
    path_dict = {}
    total_files = 0
    deleted_files = 0
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=int(mp.cpu_count())
    ) as executor:
        with tqdm.tqdm(total=len(matched_meta_data_files), smoothing=0.0) as pbar:
            for result, total, deleted, file_tuples in executor.map(
                process_files, file_tuples
            ):
                if not result:
                    failed_jobs.append(file_tuples)
                total_files += total
                deleted_files += deleted
                pbar.update(1)
                pbar.set_description(f"Total: {total_files} Deleted: {deleted_files}")

    logging.info("Done")
    logging.debug(f"Jobs failed {pprint.pformat(failed_jobs)}")

    sys.exit(0)
