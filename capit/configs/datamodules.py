from dataclasses import dataclass
from typing import Any

from capit.base.utils.typing_utils import get_module_import_path
from capit.configs.base import DataLoaderConfig
from capit.configs.datasets import InstagramImageTextMultiModalDatasetConfig
from capit.configs.string_variables import DATASET_DIR
from capit.data.datamodules import InstagramImageTextDataModule
from hydra_zen import hydrated_dataclass


@hydrated_dataclass(target=InstagramImageTextDataModule)
class InstagramImageTextMultiModalDataModuleConfig:
    dataset_config: Any = InstagramImageTextMultiModalDatasetConfig(
        dataset_dir=DATASET_DIR
    )
    data_loader_config: Any = DataLoaderConfig()
    shuffle_train: bool = True
    num_episodes_train: int = 10000000
    num_episodes_val: int = 100
    num_episodes_test: int = 100
