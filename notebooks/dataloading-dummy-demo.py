from hydra.utils import instantiate
from rich import print
from rich.traceback import install
from torch import Tensor

from capit.configs.base import DataLoaderConfig, DatasetDirectoryConfig
from capit.configs.datasets import DummyDatasetConfig

install(show_locals=False, extra_lines=1, word_wrap=True, width=350)


dataset_path_config = DatasetDirectoryConfig()
dataset_config = DummyDatasetConfig(
    dataset_dir_config=dataset_path_config, num_samples=16
)
dataloader_config = DataLoaderConfig(batch_size=16)
dataset = instantiate(config=dataset_config, _recursive_=False)
dataloader = instantiate(config=dataloader_config, dataset=dataset, _recursive_=False)

for idx, sample in enumerate(dataloader):
    for key, value in sample.items():
        if isinstance(value, Tensor):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {value}")
    if idx > 16:
        break
