from dataclasses import dataclass
from typing import Any, Optional

from hydra_zen import builds, hydrated_dataclass

from capit.base.utils.typing_utils import get_module_import_path
from capit.configs.base import DatasetConfig
from capit.data.datasets import (
    ChallengeSamplesSourceTypes,
    DummyMultiModalDataset,
    InstagramImageTextMultiModalDatasePyArrow,
    InstagramImageTextMultiModalDataset,
    SplitType,
)

DummyMultiModalDatasetConfig = DummyMultiModalDataset.default_config

InstagramImageTextMultiModalDatasetConfig = (
    InstagramImageTextMultiModalDataset.default_config
)

InstagramImageTextMultiModalDatasePyArrowConfig = (
    InstagramImageTextMultiModalDatasePyArrow.default_config
)


@dataclass
class Mode:
    fit: bool = True
    test: bool = True
    predict: bool = True
