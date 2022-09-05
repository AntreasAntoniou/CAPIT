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
import subprocess
import sys
from collections import defaultdict
from typing import Tuple

import tqdm as tqdm
from rich.logging import RichHandler

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


def convert_video_to_frames(path_tuple: Tuple[pathlib.Path, pathlib.Path]):
    video_filepath, output_dir = path_tuple
    video_filepath_string = os.fspath(video_filepath.resolve())
    output_dir_string = os.fspath(output_dir.resolve())

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    return_code = 0

    if pathlib.Path(f"{output_dir_string}".replace(".frames", ".mp4")).exists():

        command_string = [
            f"ffmpeg",
            f"-hide_banner",
            f"-loglevel",
            f"error",  # if log.level >= logging.DEBUG else "quiet",
            f"-i",
            f"{video_filepath_string}",
            f"-r",
            f"8/1",
            f"-qscale:v",
            f"4",
            f"-vf",
            f"scale=320:-1",
            f"{output_dir_string}/{video_filepath.stem}_%04d.jpg",
        ]

        process = subprocess.Popen(command_string, stdout=None, stderr=None, stdin=None)

        out, err = process.communicate(None)

        return_code = process.poll()
        if return_code != 0:
            log.exception(f"Error converting file {video_filepath_string}")
        else:
            delete_file_if_exists(
                path=pathlib.Path(f"{output_dir_string}".replace(".frames", ".mp4")),
                verbose=False,
            )

    return video_filepath_string, return_code


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

    target_file_types = [".mp4"]

    failed_jobs = []
    matching_files = defaultdict(list)
    # get all of the source audio filenames
    log.info(f"Current working directory is {args.source_filepath}")
    with tqdm.tqdm(total=4700000) as pbar:
        for file_type in target_file_types:
            for filepath in pathlib.Path(args.source_filepath).glob(f"**/*{file_type}"):
                source_filepath_string = os.fspath(filepath.resolve())
                target_folderpath_string = source_filepath_string.replace(
                    args.source_filepath, args.target_filepath
                )
                target_folderpath_string = target_folderpath_string.replace(
                    ".mp4", ".frames"
                )
                matching_files[file_type].append(
                    (filepath, pathlib.Path(target_folderpath_string))
                )
                pbar.update(1)

    for file_type in target_file_types:
        random.shuffle(matching_files[file_type])

    for file_type in target_file_types:
        num_samples = len(matching_files[file_type])
        log.info(f"Converting {num_samples}  {file_type} files to jpeg frames")
        target_func = convert_video_to_frames
        with tqdm.tqdm(total=num_samples, smoothing=0.0) as pbar:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.num_processes
            ) as executor:
                for job_idx, (video_filepath_string, return_code) in enumerate(
                    executor.map(target_func, matching_files[file_type]),
                    start=1,
                ):
                    if return_code != 0:
                        failed_jobs.append(video_filepath_string)
                    pbar.update(1)

    log.info("Done")
    log.error(f"Jobs failed {pprint.pformat(failed_jobs)}")

    sys.exit(0)
