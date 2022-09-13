from dataclasses import dataclass
from typing import Any, Optional

from capit.base.utils.typing_utils import get_module_import_path
from capit.configs.base import DatasetConfig
from capit.data.datasets import (
    ChallengeSamplesSourceTypes,
    DummyMultiModalDataset,
    InstagramImageTextMultiModalDataset,
    SplitType,
)


@dataclass
class DummyDatasetConfig(DatasetConfig):
    _target_: Any = get_module_import_path(DummyMultiModalDataset)


@dataclass
class InstagramImageTextMultiModalDatasetConfig:
    dataset_dir: str
    _target_: Any = get_module_import_path(InstagramImageTextMultiModalDataset)
    set_name: str = SplitType.TRAIN
    reset_cache: bool = False
    num_episodes: int = 10000000
    limit_num_samples: Optional[int] = None
    image_transforms: Optional[Any] = (None,)
    text_transforms: Optional[Any] = (None,)
    max_num_collection_images_per_episode: int = 1
    max_num_query_images_per_episode: int = 100
    query_image_source: str = ChallengeSamplesSourceTypes.WITHIN_USER


@dataclass
class Mode:
    fit: bool = True
    test: bool = True
    predict: bool = True
