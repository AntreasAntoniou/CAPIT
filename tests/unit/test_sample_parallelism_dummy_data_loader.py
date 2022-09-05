import logging
import time
from time import sleep

import pytest
import tqdm
from rich.logging import RichHandler
from tali.datasets.dataloaders import SampleParallelismDataLoader
from torch.utils.data import default_collate

from tali.config_repository import DatasetConfig, ImageShape, ModalityConfig
from tali.datasets.datasets import DummyMultiModalDataset

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
ch = RichHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter("%(levelname)s - %(message)s")

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
log.addHandler(ch)


@pytest.mark.parametrize("batch_size", [64])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("num_workers", [1024])
@pytest.mark.parametrize("prefetch_factor", [64])
@pytest.mark.parametrize("num_samples", [1000])
def test_data_loader(batch_size, shuffle, num_workers, prefetch_factor, num_samples):
    dataset = DummyMultiModalDataset(
        config=DatasetConfig(
            modality_config=ModalityConfig(),
            num_video_frames_per_datapoint=10,
            num_audio_frames_per_datapoint=88200,
            num_audio_sample_rate=44100,
            image_shape=ImageShape(channels=3, width=224, height=224),
            text_context_length=77,
        ),
    )
    collate_fn = default_collate
    dataloader = SampleParallelismDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_fn,
        persistent_workers=False,
        drop_last=True,
    )
    print(f"\nStart sampling {len(dataloader)} batches\n")
    start_time = time.time()
    with tqdm.tqdm(total=len(dataloader), smoothing=0.0) as pbar:
        for i, batch in enumerate(dataloader):
            current_time = time.time() - start_time
            average_batch_time = current_time / (i + 1)
            sleep(0.1)

            pbar.update(1)
            pbar.set_description(f"average_batch_time: {average_batch_time:.2f}")


#
