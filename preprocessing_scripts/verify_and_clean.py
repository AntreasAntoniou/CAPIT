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

import cv2
import numpy as np
import tqdm

from gate.utils.logging_helpers import get_logging
from tali.datasets.utils.audio import load_to_tensor


def get_base_arguments():
    parser = argparse.ArgumentParser()
    # data and I/O
    parser.add_argument("--source_filepath", type=str, default="data/")
    parser.add_argument("--num_processes", type=int, default=mp.cpu_count())

    return parser.parse_args()


def delete_file_if_exists(path: pathlib.Path):
    logging.error(f"Deleting {path}")
    if path.exists():
        path.unlink()


def verify_video(path: pathlib.Path):
    video_filepath = os.fspath(path.resolve())
    vid_capture = cv2.VideoCapture(video_filepath)
    try:
        total_frames = vid_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = vid_capture.get(cv2.CAP_PROP_FPS)
        duration_in_seconds = total_frames / fps
        vid_capture.release()
        result = True

    except Exception:
        video_path = pathlib.Path(video_filepath)
        audio_path = video_path.with_suffix(".aac")
        vid_capture.release()
        delete_file_if_exists(video_path)
        delete_file_if_exists(audio_path)
        result = False

    return video_filepath, result


def verify_audio(path: pathlib.Path):
    audio_filepath = os.fspath(path.resolve())
    try:
        load_to_tensor(
            filename=audio_filepath,
            start_point_in_seconds=1,
            duration_in_seconds=7,
            sample_rate=44100,
            mono=False,
            normalize=False,
            in_type=np.float32,
            out_type=np.float32,
            log_time=False,
            video_frame_idx_list=None,
        )
        result = True

    except Exception:
        result = False

    if not result:
        delete_file_if_exists(path)

    return audio_filepath, result


def verify_pairs(path: pathlib.Path):
    if ".mp4" in path.suffixes:
        pair_path = path.with_suffix(".aac")
    elif ".aac" in path.suffixes:
        pair_path = path.with_suffix(".mp4")
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

    target_file_types = (".mp4", ".aac")

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
        target_func = verify_video if file_type == ".mp4" else verify_audio
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.num_processes
        ) as executor:
            with tqdm.tqdm(total=num_samples, smoothing=0.0) as pbar:
                for job_idx, (file_path, result) in enumerate(
                    executor.map(target_func, matching_files[file_type]),
                    start=1,
                ):
                    pbar.update(1)

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
