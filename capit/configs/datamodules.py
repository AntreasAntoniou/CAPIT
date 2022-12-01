from dataclasses import dataclass
from typing import Any

from hydra_zen import builds, hydrated_dataclass
from torchvision.transforms import Compose, RandomCrop, Resize, ToTensor

from capit.base.utils.typing_utils import get_module_import_path
from capit.configs.base import DataLoaderConfig
from capit.configs.datasets import (
    InstagramImageTextMultiModalDatasePyArrowConfig,
    InstagramImageTextMultiModalDatasetConfig,
)
from capit.configs.string_variables import DATASET_DIR
from capit.data.datamodules import InstagramImageTextDataModule


@hydrated_dataclass(target=InstagramImageTextDataModule)
class InstagramImageTextMultiModalDataModuleConfig:
    dataset_config: Any = InstagramImageTextMultiModalDatasePyArrowConfig(
        dataset_dir=DATASET_DIR,
        image_transforms=None,
        top_k_percent="${top_percent_to_keep}",
        max_num_collection_images_per_episode="${max_num_collection_images}",
        max_num_query_images_per_episode="${max_num_challenge_images}",
    )
    data_loader_config: Any = DataLoaderConfig()
    shuffle_train: bool = True
    num_episodes_train: int = 10000000
    num_episodes_val: int = 100
    num_episodes_test: int = 100
