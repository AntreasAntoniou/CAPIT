import concurrent.futures
import itertools
import logging
import multiprocessing as mp
import os
import pathlib
import random
import re
from typing import Callable, Dict, List, Union

import h5py
import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset

from preprocessing_scripts.convert_audiofiles_to_npz import path_to_string
from tali.base import utils
from tali.config_repository import DatasetConfig
from tali.datasets.utils.audio import prevent_error_kill
from tali.datasets.utils.helpers import (
    collect_files,
    timeout,
)
from tali.datasets.utils.text import get_text_tokens
from tali.datasets.utils.video import load_frames
from tali.utils.arg_parsing import DictWithDotNotation

log = utils.get_logger(__name__)


class TALIMultiModalDataset(Dataset):
    def __init__(
        self,
        config: DatasetConfig,
        set_name: str,
        transforms: Dict[str, Union[List[Callable], Callable]],
        start_index: int = 0,
        num_samples: int = None,
    ):
        super(TALIMultiModalDataset, self).__init__()

        self.config = config
        self.set_name = set_name
        self.dataset_dir = getattr(config.dataset_dir_config, set_name)
        self.num_youtube_video_dict = {"train": 141468, "val": 6369, "test": 6500}
        self.transforms = transforms
        self.postfix = "full_video_360p"
        self.meta_data_filename = "meta_data.json"
        self.caption_data_filename = "start_timestamp_to_caption_dict_fast.json"
        if config.using_pre_sampled_split:
            self.percentage_to_keep = 1.0
        else:
            self.percentage_to_keep = (
                {
                    "milli": 0.001,
                    "centi": 0.01,
                    "deci": 0.1,
                    "base": 1.0,
                }[self.config.dataset_size_identifier]
                if set_name == "train"
                else 1.0
            )

        self.start_index = start_index
        self.hdf5_dataset_path = (
            f"{os.environ.get('PATH_CACHE_DIR')}/"
            f"{set_name}-{self.config.dataset_size_identifier}-tali.hdf5"
        )

        logging.info(
            f"{self.hdf5_dataset_path}, "
            f"{self.config.dataset_size_identifier}, "
            f"{self.percentage_to_keep}"
        )
        temp_filepath = pathlib.Path(self.hdf5_dataset_path)
        logging.debug(
            f"{self.config.rescan_paths == True and temp_filepath.exists()} "
            f"{self.config.rescan_paths} {temp_filepath.exists()}"
        )
        if (
            self.config.rescan_paths is True
            and pathlib.Path(self.hdf5_dataset_path).exists()
        ):
            # set the cache path to the experiment directory since
            # the dataset is read only and would cause issues when saving

            pathlib.Path(self.hdf5_dataset_path).unlink(missing_ok=True)

        if not pathlib.Path(self.hdf5_dataset_path).exists():
            path_dict = self._scan_paths_return_dict(
                percentage_to_keep=self.percentage_to_keep
            )

            folder_keys = list(path_dict.keys())

            dataset_file = h5py.File(self.hdf5_dataset_path, "w")

            with tqdm.tqdm(total=len(path_dict)) as pbar:
                for folder_key in folder_keys:
                    folder_key = folder_key.replace(self.dataset_dir, "")
                    prefix = f"{self.dataset_dir}/{folder_key}".replace("//", "/")
                    folder_list = path_dict[folder_key]
                    for media_tuple in folder_list:
                        (
                            frame_list,
                            video_filepath,
                            audio_filepath,
                            meta_data_filepath,
                        ) = media_tuple
                        video_filepath = video_filepath.replace(prefix, "")
                        frame_list = [
                            frame.replace(prefix, "")
                            .replace(video_filepath, "")
                            .replace(self.postfix, "")
                            .replace("/", "")
                            for frame in frame_list
                        ]
                        dataset_file.create_dataset(
                            name=f"{folder_key}/{video_filepath}",
                            data=frame_list,
                        )
                    path_dict.pop(folder_key)
                    pbar.update(1)

                dataset_file.close()

        self.dataset_file = h5py.File(self.hdf5_dataset_path, "r")

        self.video_keys = list(self.dataset_file.keys())

        self.num_samples = (
            num_samples or self.num_youtube_video_dict[self.set_name] * 100
        )

        logging.info(
            f"👍 Loaded {self.set_name} set with: \n"
            f"📊 num video videos: "
            f"{len(self.video_keys)} and "
            f"with sampler length of {self.num_samples} \n"
            f"sampled from num video clips: {len(self.video_keys)}"
        )

    @prevent_error_kill
    @timeout(60)
    def __getitem__(self, index):
        index = self.start_index + index
        actual_index = index % len(self.video_keys)
        rng = np.random.RandomState(index)
        torch_rng = torch.Generator()
        torch_rng.manual_seed(index)

        video_key = self.video_keys[actual_index]
        # log.info(f"📥 Loading video {video_key}")
        video_tuples = self.dataset_file[video_key]

        sub_video_key = rng.choice(list(video_tuples.keys()))

        # log.info(f"video_filepath_idx: {sub_video_key}")
        frame_list = video_tuples[sub_video_key]

        path_prefix = f"{self.dataset_dir}/{video_key}".replace("//", "/")
        video_filepath = f"{path_prefix}/{sub_video_key}".replace("//", "/")
        audio_filepath = f"{video_filepath}".replace(".frames", ".npz")
        meta_data_filepath = f"{path_prefix}/{self.meta_data_filename}".replace(
            "//", "/"
        )
        # log.info(f"📥 Loading {video_filepath} {audio_filepath}")
        # log.debug(
        #     f"{video_filepath} {frame_list[0]} {audio_filepath} {meta_data_filepath}"
        # )

        audio_filepath = pathlib.Path(audio_filepath)
        video_segment_idx = int(
            re.match(
                r"full_video_360p(.*).frames", video_filepath.split("/")[-1]
            ).groups()[0]
        )

        total_frames = len(frame_list)
        fps = 8
        duration_in_seconds = total_frames / float(fps)

        start_time_relative_to_full_video = int(video_segment_idx * 10)

        data_dict = DictWithDotNotation()
        data_dict.text = None
        data_dict.video = None
        data_dict.audio = None
        data_dict.image = None
        # write script that cleans up clips without any captions, write script that
        # cleans something in video space (removes videos smaller than 10 seconds)

        if self.config.modality_config.text:
            data_dict.text = self.get_text_data_tensors(
                rng,
                meta_data_filepath,
                start_time_relative_to_full_video,
                duration_in_seconds,
            )

            if data_dict.text is not None:
                data_dict.text = self.apply_transforms_if_available(
                    modality_name="text", data=data_dict.text
                )

                data_dict.text = data_dict.text.type(torch.int32)

        frames_dict = self.get_frames(
            video_filepath=video_filepath,
            frame_list=frame_list,
            audio_filepath=audio_filepath,
            meta_data_filepath=meta_data_filepath,
            rng=rng,
        )

        if frames_dict is not None:
            data_dict.update(frames_dict)

        if self.config.modality_config.video and data_dict.video is not None:
            data_dict.video = self.apply_transforms_if_available(
                modality_name="video", data=data_dict.video
            )

            data_dict.video = data_dict.video.type(torch.float32)

        if self.config.modality_config.image and data_dict.image is not None:
            data_dict.image = self.apply_transforms_if_available(
                modality_name="image", data=data_dict.image
            )

            data_dict.image = data_dict.image.type(torch.float32)

        if self.config.modality_config.audio and data_dict.audio is not None:
            data_dict.audio = self.apply_transforms_if_available(
                modality_name="audio", data=data_dict.audio
            )
            data_dict.audio = data_dict.audio.type(torch.float32)

        data_dict = {
            key: value
            for key, value in data_dict.items()
            if isinstance(value, torch.Tensor)
        }

        if self.config.modality_config.image and "image" not in data_dict:
            return None

        if self.config.modality_config.video and "video" not in data_dict:
            return None

        if self.config.modality_config.audio and "audio" not in data_dict:
            return None

        if self.config.modality_config.text and "text" not in data_dict:
            return None

        data_dict["filepath"] = video_filepath

        return data_dict

    def __len__(self):
        # use 25000 to keep training very long to ensure even val
        # intervals no matter what the size of the dataset
        return self.num_samples

    def get_frames(
        self,
        frame_list,
        video_filepath,
        audio_filepath,
        meta_data_filepath,
        rng,
    ):

        frames_dict = DictWithDotNotation()

        frames_dict.video = None
        frames_dict.audio = None
        frames_dict.image = None

        if self.config.modality_config.video:

            num_frames_to_sample_for_video = self.config.num_video_frames_per_datapoint
            # log.info(f"{len(frame_list)}, {num_frames_to_sample_for_video}")
            if len(frame_list) < num_frames_to_sample_for_video:
                selected_frame_list_idx = sorted(
                    list(
                        rng.choice(
                            len(frame_list),
                            size=(num_frames_to_sample_for_video,),
                            replace=True,
                        )
                    )
                )
            else:
                selected_frame_list_idx = sorted(
                    list(
                        rng.choice(
                            len(frame_list),
                            size=(num_frames_to_sample_for_video,),
                            replace=False,
                        )
                    )
                )

            frames_dict.video = load_frames(
                image_height=self.config.image_shape.height,
                image_width=self.config.image_shape.width,
                image_channels=self.config.image_shape.channels,
                selected_frame_list=[
                    f"{video_filepath}/"
                    f"{self.postfix}{frame_list[idx].decode('utf-8')}".replace(
                        "//", "/"
                    )
                    for idx in selected_frame_list_idx
                ],
            )

            # if frames_dict.video is None:
            #     shutil.rmtree(video_filepath)

        if self.config.modality_config.audio:
            if not pathlib.Path(audio_filepath).exists():
                return None

            frames_dict.audio = list(
                np.load(
                    audio_filepath,
                    allow_pickle=True,
                ).values()
            )[0]
            frames_dict.audio = torch.Tensor(frames_dict.audio).type(torch.float16)
            frames_dict.audio = frames_dict.audio.permute([1, 0])

            # if frames_dict.audio is None:
            #     shutil.rmtree(video_filepath)

        if self.config.modality_config.image:
            frames_dict.image = load_frames(
                image_height=self.config.image_shape.height,
                image_width=self.config.image_shape.width,
                image_channels=self.config.image_shape.channels,
                selected_frame_list=[
                    f"{video_filepath}/"
                    f"{self.postfix}{frame.decode('utf-8')}".replace("//", "/")
                    for frame in rng.choice(frame_list, (1,))
                ],
            )
            if frames_dict.image is not None:
                frames_dict.image = frames_dict.image[0]

            # if frames_dict.image is None:
            #     shutil.rmtree(video_filepath)

        return frames_dict

    def get_text_data_tensors(
        self,
        rng,
        meta_data_filepath,
        start_time_relative_to_full_video,
        duration_in_seconds,
    ):
        text = get_text_tokens(
            meta_data_filepath=meta_data_filepath,
            start_timestamp=start_time_relative_to_full_video,
            end_timestamp=start_time_relative_to_full_video + duration_in_seconds,
        )

        if not text:
            text = "No detected speech"
        else:
            text = [
                value.split(" ") if isinstance(value, str) else value
                for key, value in text.items()
            ]
            text = list(itertools.chain.from_iterable(text))
            text = [token.replace(" ", "") for token in text]
            text = " ".join(text) if len(text) > 1 else text

        return text

    # 25 * 10 ** 6 if self.set_name == "train" else
    def apply_transforms_if_available(self, modality_name, data):
        if self.transforms[modality_name]:
            if isinstance(self.transforms[modality_name], list):
                for transform in self.transforms[modality_name]:
                    data = transform(data)
            else:
                data = self.transforms[modality_name](data)

        return data

    def _scan_paths_return_dict(self, percentage_to_keep: float):

        logging.info(self.dataset_dir)

        matched_meta_data_files = []
        with tqdm.tqdm(
            total=self.num_youtube_video_dict[self.set_name], smoothing=0.0
        ) as pbar:
            for dir_path in pathlib.Path(self.dataset_dir).iterdir():
                cur_file = dir_path / "meta_data.json"
                try:
                    if cur_file.exists():
                        meta_data_string_path = path_to_string(cur_file).replace(
                            self.dataset_dir, ""
                        )
                        matched_meta_data_files.append(meta_data_string_path)
                except:
                    log.error(f"Error when checking if {cur_file} exists")
                pbar.update(1)

        logging.info(f"Found {len(matched_meta_data_files)} matched meta_data files")

        args = [
            (self.dataset_dir, item, percentage_to_keep)
            for item in matched_meta_data_files
        ]

        logging.info("Scanning folders for media files")
        path_dict = {}

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=int(mp.cpu_count())
        ) as executor:
            with tqdm.tqdm(total=len(matched_meta_data_files), smoothing=0.0) as pbar:
                for item in executor.map(collect_files, args):
                    if item is not None:
                        dataset_dir, video_key, media_tuples = item
                        if len(media_tuples) > 0:
                            path_dict[video_key] = media_tuples

                    pbar.update(1)
                    pbar.set_description(f"{len(path_dict)} subclips found")
        return path_dict


class DummyMultiModalDataset(Dataset):
    def __init__(
        self,
        config: DatasetConfig,
        set_name: str = "dummy",
        num_samples: int = 100,
        **kwargs,
    ):
        super(DummyMultiModalDataset, self).__init__()

        self.set_name = set_name
        self.config = config
        self.total_cache = 100
        self.num_samples = num_samples or self.total_cache

        random.seed(0)
        torch.manual_seed(0)
        torch_rng = torch.Generator()

        if self.config.modality_config.text:
            self.text = torch.randint(
                0,
                77,
                size=(
                    self.total_cache,
                    77,
                ),
                generator=torch_rng,
            ).int()

        if self.config.modality_config.video:
            self.video = (
                torch.randint(
                    low=1,
                    high=255,
                    size=(
                        self.total_cache,
                        self.config.num_video_frames_per_datapoint,
                        self.config.image_shape.channels,
                        self.config.image_shape.height,
                        self.config.image_shape.width,
                    ),
                    generator=torch_rng,
                ).float()
                / 255.0
            )

        if self.config.modality_config.audio:
            self.audio = (
                torch.randint(
                    low=1,
                    high=255,
                    size=(
                        self.total_cache,
                        2,
                        self.config.num_audio_frames_per_datapoint,
                    ),
                    generator=torch_rng,
                ).float()
                / 255.0
            )

        if self.config.modality_config.image:
            self.image = (
                torch.randint(
                    low=1,
                    high=255,
                    size=(
                        self.total_cache,
                        self.config.image_shape.channels,
                        self.config.image_shape.height,
                        self.config.image_shape.width,
                    ),
                    generator=torch_rng,
                ).float()
                / 255.0
            )

    def __getitem__(self, index):
        actual_index = index % self.total_cache

        data_dict = DictWithDotNotation()

        if self.config.modality_config.text:
            data_dict.text = self.text[actual_index]

        if self.config.modality_config.image:
            data_dict.image = self.image[actual_index]

        if self.config.modality_config.video:
            data_dict.video = self.video[actual_index]

        if self.config.modality_config.audio:
            data_dict.audio = self.audio[actual_index]

        data_dict.filepath = f"{self.set_name}-{index}-{actual_index}"

        return data_dict

    def __len__(self):
        return self.num_samples
