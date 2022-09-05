import inspect
import logging
import os
import pathlib
import subprocess
import time
from typing import Any, Union

import numpy as np
import torch
from torchaudio.backend.sox_io_backend import save

log = logging.getLogger(__name__)


def timeit(method):
    def timed(*args, **kwargs):
        ts = time.time()
        result = method(*args, **kwargs)
        te = time.time()

        if "log_time" in kwargs:
            if kwargs["log_time"] == True:
                log.info(f"{method.__name__} took {te - ts:.4f} sec")
        return result

    return timed


def prevent_error_kill(method):
    def try_catch_return(*args, **kwargs):
        try:
            result = method(*args, **kwargs)
            return result
        except Exception as e:
            log.exception(f"{method.__name__} error: {e}")
            return None

    return try_catch_return


class AudioLoadingError(Exception):
    """Base class for exceptions in this module."""

    pass


# load_audio can not detect the input type
def load_to_tensor(
    filename: str,
    sample_rate: int = 44100,
    mono: bool = False,
    in_type=np.float32,
    out_type=np.float32,
):
    channels = 1 if mono else 2
    format_strings = {
        np.float64: "f64le",
        np.float32: "f32le",
        np.int16: "s16le",
        np.int32: "s32le",
        np.uint32: "u32le",
    }
    format_string = format_strings[in_type]
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error" if log.level >= logging.DEBUG else "quiet",
        "-i",
        filename,
        "-f",
        format_string,
        "-acodec",
        f"pcm_{format_string}",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=None, stdin=None)
    out, err = process.communicate(None)

    retcode = process.poll()

    if retcode != 0:
        log.exception(f"Error loading audio file {filename}")
        raise AudioLoadingError(
            f"{inspect.stack()[0][3]} returned non-zero exit code {retcode}"
        )

    audio = np.frombuffer(out, dtype=in_type).astype(out_type)

    audio = audio.reshape(-1, channels)

    audio = torch.Tensor(audio)

    return audio


def convert_audiofile_to_tensor(
    filepath: Union[str, pathlib.Path],
    sample_rate: int = 44100,
    mono: bool = False,
    in_type: Any = np.float32,
    out_type: Any = np.float32,
):
    if isinstance(filepath, pathlib.Path):
        filepath = os.fspath(filepath.resolve())

    if pathlib.Path(filepath).exists():

        return load_to_tensor(
            filepath,
            sample_rate=sample_rate,
            mono=mono,
            in_type=in_type,
            out_type=out_type,
        ).numpy()

    return None


def tensor_to_audio(
    input_audio: torch.Tensor,
    output_path: pathlib.Path,
    format: str = "wav",
    sample_rate: int = 44100,
):
    try:
        # input_audio = input_audio.permute([1, 0])

        save(
            output_path.with_suffix(f".{format}"),
            input_audio,
            sample_rate,
            format=format,
        )
        return True
    except Exception:
        log.exception(
            f"Converting tensor to an audio file failed on file "
            f"{output_path.with_suffix(f'.{format}')}"
        )
        return False
