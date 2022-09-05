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
import sys
from collections import defaultdict
from typing import Tuple

import numpy as np
import tqdm as tqdm
from rich.logging import RichHandler

from tali.datasets.utils.audio import convert_audiofile_to_tensor

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


class VideoToFrameError(Exception):
    """Base class for exceptions in this module."""

    pass


class ConversionFailedError(VideoToFrameError):
    pass


def get_base_arguments():
    parser = argparse.ArgumentParser()
    # data and I/O
    parser.add_argument("--source_filepath", type=str, default="source_data/")
    parser.add_argument("--target_filepath", type=str, default="target_data/")
    parser.add_argument("--num_processes", type=int, default=mp.cpu_count())

    return parser.parse_args()


path_to_string = lambda x: os.fspath(x.resolve())


def convert_audiofile_to_npz(path_tuple: Tuple[pathlib.Path, pathlib.Path]):
    source_audio_filepath, target_numpy_filepath = path_tuple
    try:
        if not target_numpy_filepath.parent.exists():
            target_numpy_filepath.parent.mkdir(parents=True)

        audio_array = convert_audiofile_to_tensor(
            path_to_string(source_audio_filepath),
            sample_rate=44100,
            mono=False,
            in_type=np.float32,
            out_type=np.float32,
        )
        audio_array = audio_array.astype(np.float16)

        if audio_array is None:
            log.exception(f"Error converting file {audio_filepath_string}")
            return source_audio_filepath, False
        else:
            np.savez_compressed(
                path_to_string(target_numpy_filepath), audio_array=audio_array
            )
            delete_file_if_exists(
                path=source_audio_filepath,
                verbose=False,
            )
            return source_audio_filepath, True
    except Exception:
        return source_audio_filepath, False


def delete_file_if_exists(path: pathlib.Path, verbose: bool = True):

    if path.exists():
        if verbose:
            log.info(f"Deleting {path}")
        path.unlink()


if __name__ == "__main__":
    args = get_base_arguments()

    if not os.path.exists(args.source_filepath):
        log.error(f"Source path {args.source_filepath} not found")

    if not os.path.exists(args.target_filepath):
        os.makedirs(args.target_filepath, exist_ok=True)

    target_file_types = [".aac"]

    failed_jobs = []
    matching_files = defaultdict(list)
    # get all of the source audio filenames
    log.info(f"Current working directory is {args.source_filepath}")
    with tqdm.tqdm(total=4700000) as pbar:
        for file_type in target_file_types:
            for filepath in pathlib.Path(args.source_filepath).glob(f"**/*{file_type}"):
                source_filepath_string = path_to_string(filepath)
                target_folderpath_string = source_filepath_string.replace(
                    args.source_filepath, args.target_filepath
                )
                target_folderpath_string = target_folderpath_string.replace(
                    ".aac", ".npz"
                )
                matching_files[file_type].append(
                    (filepath, pathlib.Path(target_folderpath_string))
                )
                pbar.update(1)

    for file_type in target_file_types:
        random.shuffle(matching_files[file_type])

    for file_type in target_file_types:
        num_samples = len(matching_files[file_type])
        log.info(f"Converting {num_samples}  {file_type} files to npz")
        target_func = convert_audiofile_to_npz
        with tqdm.tqdm(total=num_samples, smoothing=0.0) as pbar:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.num_processes
            ) as executor:
                for job_idx, (audio_filepath_string, return_code) in enumerate(
                    executor.map(target_func, matching_files[file_type]),
                    start=1,
                ):
                    if return_code is False:
                        failed_jobs.append(audio_filepath_string)
                    pbar.update(1)

    log.info("Done")
    log.error(f"Jobs failed {pprint.pformat(failed_jobs)}")

    sys.exit(0)
