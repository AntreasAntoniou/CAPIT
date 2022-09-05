import logging
import os
import pathlib
from collections import defaultdict
from typing import Optional

import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import wandb
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, seed_everything
from torchvision.utils import make_grid
from wandb.plots.heatmap import heatmap

from tali.datasets.tokenizers import HuggingFaceBPETokenizer
from tali.datasets.utils.audio import tensor_to_audio

log = logging.getLogger(__name__)
tokenizer = HuggingFaceBPETokenizer(context_length=77)


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


def create_storage_dir(parent_dir: pathlib.Path, set_name: str):
    store_dir = parent_dir / set_name

    if not store_dir.exists():
        store_dir.mkdir(parents=True)

    return store_dir


def make_image_frame_grid(
    image_frames: torch.Tensor,
    num_video_frames_per_datapoint: int,
    save: bool = False,
    store_dir: Optional[pathlib.Path] = None,
    filename: Optional[str] = None,
    show: bool = False,
):
    image_grid = make_grid(
        tensor=image_frames,
        normalize=True,
        value_range=None,
        scale_each=True,
        pad_value=0,
        nrow=num_video_frames_per_datapoint,
    )

    image_grid = image_grid.permute([1, 2, 0]).cpu().numpy()
    image_grid = cv2.cvtColor(image_grid, cv2.COLOR_BGR2RGB)
    plt.imshow(image_grid)
    if save and store_dir is not None:
        plt.savefig(fname=store_dir / f"{filename}.png")
    if show:
        plt.show()
    return image_grid


def decode_and_store_text(
    text_frames: torch.Tensor,
    save: bool = False,
    store_dir: Optional[pathlib.Path] = None,
    filename: Optional[str] = None,
    show: bool = False,
):

    decoded_text = tokenizer.batch_decode(x=text_frames)
    decoded_text = [text.replace("</w>", "").replace("!", "") for text in decoded_text]

    if save:
        output_text_dir = store_dir / f"{filename}.txt"
        with open(output_text_dir, mode="w") as file_writer:
            file_writer.writelines(decoded_text)

    if show:
        log.info(decoded_text)

    return decoded_text


def decode_audio_plot_audiograph_store_audio_file(
    audio_frames: torch.Tensor,
    save_audiograph: bool = False,
    save_audio_file: bool = False,
    store_dir: Optional[pathlib.Path] = None,
    filename: Optional[str] = None,
    show: bool = False,
):
    for item_idx, audio_item in enumerate(audio_frames):
        logging.info(f"{torch.mean(audio_item), torch.std(audio_item)}, ")

        for channel in range(2):
            fig = plot_with_spectrum(audio_item[channel], rate=44100)
            if save_audiograph:
                fig.savefig(
                    store_dir
                    / f"{filename}_{item_idx}_audiograph_channel_{channel}.txt"
                )
            if show:
                plt.show()
            plt.close()

        if save_audio_file:
            tensor_to_audio(
                input_audio=audio_item,
                output_path=pathlib.Path(store_dir / f"{filename}_{item_idx}.wav"),
            )
            return pathlib.Path(store_dir / f"{filename}_{item_idx}.wav")


def sample_and_upload_pred_heatmap(config: DictConfig):
    seed_everything(config.seed, workers=True)

    log.info("Start uploading 😼")

    datamodule: LightningDataModule = hydra.utils.instantiate(
        config.datamodule, _recursive_=False
    )
    datamodule.setup(stage="fit")
    datamodule.setup(stage="test")

    run = wandb.init(
        project=config.logger.wandb.project,
        name=config.logger.wandb.name,
        resume=True,
    )
    columns = ["id", "video", "image", "audio", "text"]
    modalities = ["video", "image", "audio", "text"]
    current_log_idx = 0
    dataset_dict_caller_fn = {
        "train": (datamodule.train_dataloader, datamodule.train_start_index),
        "val": (datamodule.val_dataloader, datamodule.val_start_index),
        "test": (datamodule.test_dataloader, datamodule.test_start_index),
    }
    dataset_dict_loaders = {
        key: (value[0](), value[1])
        for key, value in dataset_dict_caller_fn.items()
        if key in config.wandb_visualization_config.sets_to_upload
    }
    round_float = lambda x: (x * 10**3).round() / (10**3)
    for key, (dataloader, start_idx) in dataset_dict_loaders.items():
        multimedia_log_file = wandb.Table(columns=columns)
        current_log_idx = 0
        with tqdm.tqdm(
            initial=start_idx,
            total=config.wandb_visualization_config.num_samples_to_upload_per_set,
            smoothing=0.0,
        ) as pbar:
            for batch_idx, item_batch in enumerate(dataloader):
                if (
                    current_log_idx
                    >= config.wandb_visualization_config.num_samples_to_upload_per_set
                ):
                    break
                image_batch = item_batch["image"]
                video_batch = item_batch["video"]
                audio_batch = item_batch["audio"]
                text_batch = item_batch["text"]
                filepath_batch = item_batch["filepath"]

                text_batch = decode_and_store_text(
                    text_frames=text_batch,
                    save=False,
                    show=False,
                )
                filepath_batch = [
                    filepath.replace(os.environ.get("DATASET_DIR"), "")
                    .replace("full_video_360p", "")
                    .replace(".frames", "")
                    for filepath in filepath_batch
                ]
                rich_media_dict = defaultdict(list)
                for image, video, audio, text, filepath in zip(
                    image_batch,
                    video_batch,
                    audio_batch,
                    text_batch,
                    filepath_batch,
                ):
                    video = video.permute([0, 2, 3, 1]).cpu().numpy()
                    video_shape = video.shape
                    video = video.reshape(-1, video.shape[2], video.shape[3])
                    video = cv2.cvtColor(video, cv2.COLOR_BGR2RGB)
                    video = video.reshape(video_shape) * 255
                    video = torch.Tensor(video).permute([0, 3, 1, 2]).type(torch.uint8)

                    video_log_file = wandb.Video(video, fps=8, format="gif")

                    image_log_file = wandb.Image(
                        make_image_frame_grid(
                            image_frames=image.unsqueeze(0),
                            num_video_frames_per_datapoint=1,
                            save=False,
                            store_dir=None,
                            filename=None,
                            show=False,
                        )
                    )

                    audio_log_file = wandb.Audio(
                        audio.permute([1, 0]), sample_rate=44100
                    )

                    # /mnt/disk/tali/dataset-in-frames/
                    # val/--4ZIf4_5aU/full_video_360p0034.frames

                    rich_media_dict["video"].append(video_log_file)
                    rich_media_dict["image"].append(image_log_file)
                    rich_media_dict["audio"].append(audio_log_file)
                    rich_media_dict["text"].append(text)

                    current_log_idx += 1
                    pbar.update(1)

                log.info(f"Uploading {key}-set_chunk_{current_log_idx}")
                run.log({f"{key}-heatmap-data-{batch_idx}": multimedia_log_file})
                multimedia_log_file = wandb.Table(columns=columns)
                output_similarities = {}
                for source_modality in modalities:
                    for target_modality in modalities:

                        if (
                            f"{source_modality}-{target_modality}-preds"
                            not in output_similarities
                        ):

                            random_preds = (
                                torch.randn(size=(config.batch_size, config.batch_size))
                                .type(torch.float32)
                                .cpu()
                                .numpy()
                            )
                            random_preds = np.around(random_preds, decimals=3)
                            output_similarities[
                                f"{source_modality}-{target_modality}-preds"
                            ] = random_preds

                            output_similarities[
                                f"{target_modality}-{source_modality}-preds"
                            ] = random_preds.transpose(0, 1)

                            run.log(
                                {
                                    f"{key}-id_heatmap"
                                    f"-x={source_modality}"
                                    f"-y={target_modality}": heatmap(
                                        x_labels=rich_media_dict[source_modality],
                                        y_labels=rich_media_dict[target_modality],
                                        matrix_values=random_preds,
                                        show_text=True,
                                    )
                                }
                            )
