import multiprocessing
from dataclasses import dataclass
from typing import Any, Optional

from capit.base.utils.typing_utils import get_module_import_path
from capit.configs.string_variables import BATCH_SIZE
from hydra_zen import hydrated_dataclass
from torch.utils.data import DataLoader


@dataclass
class ImageShape:
    channels: int = 3
    width: int = 224
    height: int = 224


@dataclass
class ModalityConfig:
    image: bool = True
    audio: bool = False
    video: bool = False
    text: bool = True


@hydrated_dataclass(target=DataLoader)
class DataLoaderConfig:
    dataset: Any = None
    _target_: str = get_module_import_path(DataLoader)
    batch_size: int = BATCH_SIZE
    persistent_workers: bool = False
    pin_memory: bool = True
    prefetch_factor: int = 2
    num_workers: int = multiprocessing.cpu_count()
    shuffle: bool = True


@dataclass
class DatasetDirectoryConfig:
    train: Optional[str] = None
    val: Optional[str] = None
    test: Optional[str] = None


@dataclass
class DatasetConfig:
    _target_: Any = None
    dataset_dir_config: DatasetDirectoryConfig = DatasetDirectoryConfig()
    set_name: str = "dummy"
    num_samples: int = 100
    using_pre_sampled_split: bool = False
    dataset_size_identifier: str = "base"
    dataset_name: str = "base"
    modality_config: ModalityConfig = ModalityConfig()
    rescan_paths: bool = False
    num_video_frames_per_datapoint: int = 10
    num_audio_frames_per_datapoint: int = 88200
    num_audio_sample_rate: int = 44100
    image_shape: ImageShape = ImageShape(channels=3, width=224, height=224)
    text_context_length: int = 77
