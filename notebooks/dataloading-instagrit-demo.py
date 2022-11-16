from hydra.utils import instantiate
from rich import print
from rich.traceback import install
from torch import Tensor

from capit.configs.base import DataLoaderConfig, DatasetDirectoryConfig
from capit.configs.datasets import InstagramImageTextMultiModalDatasetConfig
from capit.data.transforms import image_transforms_base, text_transforms_base

install(show_locals=False, extra_lines=1, word_wrap=True, width=350)


dataset_path_config = DatasetDirectoryConfig()
dataset_config = InstagramImageTextMultiModalDatasetConfig(
    dataset_dir="/mnt/nas/datasets/instagram-influencers/instagram-celebrities",
    reset_cache=False,
    limit_num_samples=10000,
    image_transforms=image_transforms_base(),
    text_transforms=text_transforms_base(),
)
dataset = instantiate(config=dataset_config, _recursive_=False)
print(dataset.__dict__)
dataloader_config = DataLoaderConfig(batch_size=16)
dataloader = instantiate(config=dataloader_config, dataset=dataset, _recursive_=False)

for idx, sample in enumerate(dataloader):
    for key, value in sample.items():
        if isinstance(value, Tensor):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {value}")
    if idx > 16:
        break
