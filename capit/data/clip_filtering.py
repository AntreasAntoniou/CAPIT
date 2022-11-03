import argparse
from hydra_zen import instantiate
from rich import print
from rich.traceback import install
from yaml import parse
from capit.base.utils.tf_babysitting import configure_tf_memory_growth

from capit.data.datasets import (
    InstagramImageTextMultiModalDataset,
    InstagramImageTextMultiModalDatasetByUser,
)

from capit.models.image_text_models import CLIPImageTextModel

install()
configure_tf_memory_growth()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket_size", type=int, default=300)
    args = parser.parse_args()

    dataset_module = InstagramImageTextMultiModalDatasetByUser
    dataset_config = dataset_module.default_config
    print(dataset_module, dataset_config(dataset_dir="/data/"))
    dataset = instantiate(config=dataset_config, dataset_dir="/data/")
    model_module = CLIPImageTextModel
    model_config = model_module.default_config
    model = instantiate(config=model_config, dataset=dataset)

    image_bucket = []
    caption_bucket = []
    filepath_bucket = []

    for i, batch in enumerate(dataset):
        image_bucket.extend(batch["image"])
        caption_bucket.extend(batch["caption"])
        filepath_bucket.extend(batch["filepath"])
        if len(image_bucket) >= args.bucket_size:
            image_batch = image_bucket[: args.bucket_size]
            caption_batch = caption_bucket[: args.bucket_size]
            filepath_batch = filepath_bucket[: args.bucket_size]

            image_bucket = image_bucket[args.bucket_size :]
            caption_bucket = caption_bucket[args.bucket_size :]
            filepath_bucket = filepath_bucket[args.bucket_size :]
