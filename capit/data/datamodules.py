import functools
from dataclasses import dataclass
from typing import Optional, Any

from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS

from capit.base.utils.typing_utils import get_module_import_path
from capit.configs.base import DataLoaderConfig, DatasetConfig
from capit.configs.datasets import InstagramImageTextMultiModalDatasetConfig
from capit.data.transforms import image_transforms_base


class DataModule(LightningDataModule):
    def __init__(
        self,
        dataset_config: DictConfig,
        data_loader_config: DictConfig,
    ):
        super(DataModule, self).__init__()
        self.dataset_config = dataset_config
        self.data_loader_config = data_loader_config

    def setup(self, stage: Optional[str] = None):
        raise NotImplementedError

    def train_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        raise NotImplementedError


@dataclass
class ImageTextTransformConfig:
    image_transforms: Any = image_transforms_base()
    text_transforms: Any = None


@dataclass
class SplitType:
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class InstagramImageTextDataModule(DataModule):
    def __init__(
        self,
        dataset_config: Any,
        data_loader_config: Any,
        transform_train: ImageTextTransformConfig = ImageTextTransformConfig(),
        transform_eval: ImageTextTransformConfig = ImageTextTransformConfig(),
    ):
        super(InstagramImageTextDataModule, self).__init__(
            dataset_config=dataset_config, data_loader_config=data_loader_config
        )

        self.transform_train = transform_train
        self.transform_eval = transform_eval

    def setup(self, stage: Optional[str] = None):

        if stage == "fit" or stage is None:
            self.train_set = instantiate(
                config=self.dataset_config,
                set_name=SplitType.TRAIN,
                image_transforms=self.transform_train.image_transforms,
                text_transforms=self.transform_train.text_transforms,
                num_episodes=1000000,
            )

            self.val_set = instantiate(
                config=self.dataset_config,
                set_name=SplitType.VAL,
                image_transforms=self.transform_eval.image_transforms,
                text_transforms=self.transform_eval.text_transforms,
                num_episodes=100,
            )

        elif stage == "validate":
            self.val_set = instantiate(
                config=self.dataset_config,
                set_name=SplitType.VAL,
                image_transforms=self.transform_eval.image_transforms,
                text_transforms=self.transform_eval.text_transforms,
                num_episodes=100,
            )

        # Assign test dataset for use in dataloader(s)
        elif stage == "test":
            self.test_set = instantiate(
                config=self.dataset_config,
                set_name=SplitType.TEST,
                image_transforms=self.transform_eval.image_transforms,
                text_transforms=self.transform_eval.text_transforms,
                num_episodes=1000,
            )

        else:
            raise ValueError(f"Invalid stage name passed {stage}")

    def train_dataloader(self):

        return instantiate(
            self.data_loader_config, dataset=self.train_set, shuffle=True
        )

    def val_dataloader(self):

        return instantiate(self.data_loader_config, dataset=self.val_set, shuffle=False)

    def test_dataloader(self):

        return instantiate(
            self.data_loader_config, dataset=self.test_set, shuffle=False
        )

    def predict_dataloader(self):
        return self.test_dataloader()
