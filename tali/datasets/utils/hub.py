import logging
import os
import pathlib
import signal
from functools import wraps

import numpy as np
from torch.utils.data import dataloader

log = logging.getLogger(__name__)


def timeout(timeout_secs: int):
    def wrapper(func):
        @wraps(func)
        def time_limited(*args, **kwargs):
            # Register an handler for the timeout
            def handler(signum, frame):
                raise Exception(f"Timeout for function '{func.__name__}'")

            # Register the signal function handler
            signal.signal(signal.SIGALRM, handler)

            # Define a timeout for your function
            signal.alarm(timeout_secs)

            result = None
            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                logging.error(f"Exploded due to time out on {args, kwargs}")
                raise exc
            finally:
                # disable the signal alarm
                signal.alarm(0)

            return result

        return time_limited

    return wrapper


def collect_files(args):
    # sourcery skip: identity-comprehension, simplify-len-comparison, use-named-expression
    json_file_path, training_set_size_fraction_value = args
    video_files = list(pathlib.Path(json_file_path.parent).glob("**/*.frames"))
    video_key = json_file_path.parent.stem
    folder_list = []
    for file in video_files:
        video_data_filepath = os.fspath(file.resolve())
        frame_list = list(pathlib.Path(file).glob("**/*.jpg"))
        frame_list = [os.fspath(frame.resolve()) for frame in frame_list]

        if len(frame_list) > 0:
            frame_idx_to_filepath = {
                int(frame_filepath.split("_")[-1].replace(".jpg", "")): frame_filepath
                for frame_filepath in frame_list
            }

            frame_idx_to_filepath = {
                k: v for k, v in sorted(list(frame_idx_to_filepath.items()))
            }
            frame_list = list(frame_idx_to_filepath.values())
            audio_data_filepath = os.fspath(file.resolve()).replace(".frames", ".aac")
            meta_data_filepath = os.fspath(json_file_path.resolve())

            if (
                pathlib.Path(video_data_filepath).exists()
                and pathlib.Path(meta_data_filepath).exists()
                and pathlib.Path(audio_data_filepath).exists()
            ) and np.random.random() <= training_set_size_fraction_value:
                folder_list.append(
                    (
                        frame_list,
                        video_data_filepath,
                        audio_data_filepath,
                        meta_data_filepath,
                    )
                )

    return video_key, folder_list


def collate_resample_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return dataloader.default_collate(batch)
