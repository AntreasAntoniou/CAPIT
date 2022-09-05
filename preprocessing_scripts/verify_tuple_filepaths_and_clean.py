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
import random
import shutil
import sys
from collections import defaultdict

import tqdm

from gate.utils.logging_helpers import get_logging


def get_base_arguments():
    parser = argparse.ArgumentParser()
    # data and I/O
    parser.add_argument("--source_filepath", type=str, default="data/")
    parser.add_argument("--num_processes", type=int, default=mp.cpu_count())

    return parser.parse_args()


def delete_file_if_exists(path: pathlib.Path):
    logging.debug(f"Deleting {path}")
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        if path.is_file():
            path.unlink()


def verify_pairs(path: pathlib.Path):
    if ".frames" in path.suffixes:
        pair_path = path.with_suffix(".npz")
    elif ".npz" in path.suffixes:
        pair_path = path.with_suffix(".frames")
    else:
        return path, False

    if pair_path.exists() and path.exists():
        return path, True

    delete_file_if_exists(path)
    delete_file_if_exists(pair_path)
    return path, False


if __name__ == "__main__":
    logging = get_logging("NOTSET")
    args = get_base_arguments()

    if not os.path.exists(args.source_filepath):
        logging.error(f"Source path {args.source_filepath} not found")

    target_file_types = (".frames", ".npz")

    failed_jobs = []
    matching_files = defaultdict(list)
    # get all of the source audio filenames
    logging.info(f"Current working directory is {args.source_filepath}")
    with tqdm.tqdm() as pbar:
        for file_type in target_file_types:
            for file in pathlib.Path(args.source_filepath).glob(f"**/*{file_type}"):
                matching_files[file_type].append(file)
                pbar.update(1)

    for file_type in target_file_types:
        random.shuffle(matching_files[file_type])

    for file_type in target_file_types:
        num_samples = len(matching_files[file_type])
        logging.info(f"Checking {num_samples}  {file_type} files")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.num_processes
        ) as executor:
            with tqdm.tqdm(total=num_samples, smoothing=0.0) as pbar:
                for job_idx, (file_path, result) in enumerate(
                    executor.map(verify_pairs, matching_files[file_type]),
                    start=1,
                ):
                    pbar.update(1)

    logging.info("Done")
    logging.error(f"Jobs failed {pprint.pformat(failed_jobs)}")

    sys.exit(0)
