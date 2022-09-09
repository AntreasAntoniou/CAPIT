from dataclasses import dataclass
from typing import Any, Optional

from capit.configs.base import DataLoaderConfig

from capit.base.utils.typing_utils import get_module_import_path
from capit.configs.datasets import InstagramImageTextMultiModalDatasetConfig
from capit.configs.string_variables import DATASET_DIR
from capit.data.datamodules import (
    InstagramImageTextDataModule,
)


@dataclass
class InstagramImageTextMultiModalDataModuleConfig:
    _target_: Any = get_module_import_path(InstagramImageTextDataModule)
    dataset_config: Any = InstagramImageTextMultiModalDatasetConfig(
        dataset_dir=DATASET_DIR
    )
    data_loader_config: Any = DataLoaderConfig()
