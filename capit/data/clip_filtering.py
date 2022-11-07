import argparse
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import pathlib
from typing import List
from hydra_zen import instantiate
from rich import print
from rich.traceback import install
import torch
from torch.utils.data import DataLoader
import tqdm
from yaml import parse
from capit.base.utils.loggers import get_logger
from capit.base.utils.tf_babysitting import configure_tf_memory_growth
from capit.configs.base import DataLoaderConfig

from capit.data.datasets import (
    InstagramImageTextMultiModalDataset,
    InstagramImageTextMultiModalDatasetByUser,
    SplitType,
)

from capit.models.image_text_models import CLIPImageTextModel

install()
configure_tf_memory_growth()

logger = get_logger(set_default_handler=True)


def collate_batch(batch):

    output = defaultdict(list)
    for item in batch:
        for key, value in item.items():
            output[key].extend(value)

    return output


@dataclass
class ResponseTypes:
    DONE: int = 0
    EXISTS: int = 1
    FAILED: int = 2


import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


@dataclass
class TableEntrySchema:
    user_name: pa.string()
    id: pa.string()
    filepath: pa.list_(pa.string())
    similarity: pa.float32()


table_entry_schema = list(TableEntrySchema.__annotations__.items())
table_entry_schema = pa.schema(table_entry_schema)


def add_row_to_table(
    filepath: str,
    similarity: float,
    id: str,
    user_name: str,
) -> int:
    try:
        user_name_filepath = data_root / f"{user_name}"
        entry_filepath = user_name_filepath / f"{id}.parquet"

        if entry_filepath.exists():
            return ResponseTypes.EXISTS

        if not user_name_filepath.exists():
            user_name_filepath.mkdir(parents=True, exist_ok=True)

        table_entry = pa.table(
            [
                pa.array([user_name], type=pa.string()),
                pa.array([id], type=pa.int32()),
                pa.array([filepath], type=pa.list_(pa.string())),
                pa.array([similarity], type=pa.float32()),
            ],
            schema=table_entry_schema,
        )

        pq.write_table(table=table_entry, where=entry_filepath)

        return ResponseTypes.DONE
    except Exception as e:
        logger.exception(e)
        return ResponseTypes.FAILED


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket_size", type=int, default=1800)
    parser.add_argument("--data_root", type=str, default="/data/instagram_table/")
    args = parser.parse_args()

    data_root = pathlib.Path(args.data_root)

    if not data_root.exists():
        data_root.mkdir(parents=True, exist_ok=True)

    dataset_module = InstagramImageTextMultiModalDatasetByUser
    dataset_config = dataset_module.default_config
    print(dataset_module, dataset_config(dataset_dir="/data/"))
    train_dataset = instantiate(
        config=dataset_config, dataset_dir="/data/", set_name=SplitType.TRAIN
    )
    val_dataset = instantiate(
        config=dataset_config, dataset_dir="/data/", set_name=SplitType.VAL
    )
    test_dataset = instantiate(
        config=dataset_config, dataset_dir="/data/", set_name=SplitType.TEST
    )

    data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=4,
        pin_memory=False,
        num_workers=8,
        shuffle=False,
        collate_fn=collate_batch,
    )

    model_module = CLIPImageTextModel
    model_config = model_module.default_config
    model = instantiate(
        config=model_config,
        model_name_or_path="openai/clip-vit-large-patch14",
    )
    model.to(torch.cuda.current_device())

    image_bucket = []
    caption_bucket = []
    filepath_bucket = []
    user_name_bucket = []
    ids_bucket = []
    logits_bucket = []
    with tqdm.tqdm(total=len(data_loader), smoothing=0.0) as pbar:
        for i, batch in enumerate(data_loader):
            image_bucket.extend(batch["image"])
            caption_bucket.extend(batch["text"])
            filepath_bucket.extend(batch["filepath"])
            user_name_bucket.extend(batch["user_name"])
            ids_bucket.extend(batch["ids"])

            while len(image_bucket) >= args.bucket_size:
                image_batch = image_bucket[: args.bucket_size]
                caption_batch = caption_bucket[: args.bucket_size]
                filepath_batch = filepath_bucket[: args.bucket_size]
                user_name_batch = user_name_bucket[: args.bucket_size]
                ids_bucket_batch = ids_bucket[: args.bucket_size]

                user_name_filepath = table_filepath = (
                    data_root / f"{user_name_batch[0]}"
                )

                image_bucket = image_bucket[args.bucket_size :]
                caption_bucket = caption_bucket[args.bucket_size :]
                filepath_bucket = filepath_bucket[args.bucket_size :]
                user_name_bucket = user_name_bucket[args.bucket_size :]
                ids_bucket = ids_bucket[args.bucket_size :]

                if user_name_filepath.exists():
                    continue

                with torch.no_grad():
                    similarity_batch = model.predict_individual(
                        image=image_batch, text=caption_batch
                    )
                    similarity_batch = similarity_batch.detach().cpu().tolist()

                for filepath, similarity, idx, user_name in zip(
                    filepath_batch, similarity_batch, ids_bucket_batch, user_name_batch
                ):
                    response = add_row_to_table(
                        filepath=list(filepath),
                        similarity=similarity,
                        id=idx,
                        user_name=user_name,
                    )

            pbar.update(1)
