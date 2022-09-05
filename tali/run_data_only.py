import logging

import hydra
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, seed_everything

from tali.utils.storage import pretty_print_dict

log = logging.getLogger(__name__)


def plot_with_spectrum(x, rate=48000):
    """Plot the given waveform (timeSeries), both as time-domain and as its
    frequency-domain spectrum. Returns a matplotlib.figure.Figure object."""
    fig, axs = plt.subplots(2)
    n = len(x)
    # (1) Plot time-domain data:
    timesMsec = np.arange(n) * 1000.0 / rate
    axs[0].plot(timesMsec, x)
    # Limit the X axis to our input samples:
    axs[0].set_xlabel("Time (ms)")
    axs[0].grid(True)
    # (2) Compute and plot frequency spectrum:
    return fig


def sample_datamodule(config: DictConfig):
    seed_everything(config.seed, workers=True)

    log.info(f"{pretty_print_dict(dict(config))}")

    datamodule: LightningDataModule = hydra.utils.instantiate(
        config.datamodule, _recursive_=False
    )
    datamodule.setup(stage="fit")
    datamodule.setup(stage="test")

    with tqdm.tqdm(total=len(datamodule.val_dataloader()), smoothing=0.0) as pbar:
        for idx, item_batch in enumerate(datamodule.val_dataloader()):
            pbar.update(1)
            # text_batch = decode_and_store_text(
            #     text_frames=item_batch["text"],
            #     save=False,
            #     show=False,
            # )
            # pbar.set_description(f"{text_batch}")

    with tqdm.tqdm(total=len(datamodule.test_dataloader()), smoothing=0.0) as pbar:
        for idx, item_batch in enumerate(datamodule.test_dataloader()):
            pbar.update(1)
            # text_batch = decode_and_store_text(
            #     text_frames=item_batch["text"],
            #     save=False,
            #     show=False,
            # )
            # pbar.set_description(f"{text_batch}")

    with tqdm.tqdm(total=len(datamodule.train_dataloader()), smoothing=0.0) as pbar:
        for idx, item_batch in enumerate(datamodule.train_dataloader()):
            pbar.update(1)
            # text_batch = decode_and_store_text(
            #     text_frames=item_batch["text"],
            #     save=False,
            #     show=False,
            # )
            # pbar.set_description(f"{text_batch}")
