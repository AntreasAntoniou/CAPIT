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

import tqdm

from gate.utils.logging_helpers import get_logging
from tali.datasets.utils.helpers import load_text_into_language_time_stamps


def get_base_arguments():
    parser = argparse.ArgumentParser()
    # data and I/O
    parser.add_argument("--source_filepath", type=str, default="data/")
    parser.add_argument("--num_processes", type=int, default=mp.cpu_count())

    return parser.parse_args()


def delete_file_if_exists(path: pathlib.Path):
    logging.error(f"Delete {path}")
    if path.exists():
        path.unlink()


def verify_meta_data_and_generate_caption_json(path: pathlib.Path):
    filepath = os.fspath(path.resolve())
    folderpath = os.fspath(path.parent.resolve())
    if path.exists():
        try:
            load_text_into_language_time_stamps(filepath=filepath)
            return path, True
        except Exception:
            delete_file_if_exists(path)
            shutil.rmtree(folderpath)

    return path, False


if __name__ == "__main__":
    logging = get_logging("NOTSET")
    args = get_base_arguments()

    if not os.path.exists(args.source_filepath):
        logging.error(f"Source path {args.source_filepath} not found")

    target_file_types = "meta_data.json"

    failed_jobs = []
    matching_files = []
    # get all of the source audio filenames
    logging.info(f"Current working directory is {args.source_filepath}")
    with tqdm.tqdm() as pbar:
        for file in pathlib.Path(args.source_filepath).glob("**/meta_data.json"):
            matching_files.append(file)
            pbar.update(1)

    num_samples = len(matching_files)
    logging.info(f"Checking {num_samples} {target_file_types} files")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.num_processes
    ) as executor:
        with tqdm.tqdm(total=num_samples, smoothing=0.0) as pbar:
            for job_idx, (file_path, result) in enumerate(
                executor.map(
                    verify_meta_data_and_generate_caption_json, matching_files
                ),
                start=1,
            ):
                pbar.update(1)

    logging.info("Done")
    logging.error(f"Jobs failed {pprint.pformat(failed_jobs)}")

    sys.exit(0)
