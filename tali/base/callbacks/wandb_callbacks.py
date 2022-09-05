import os
from pathlib import Path
from typing import List, Dict, Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import torch
import wandb
from pytorch_lightning import Callback, Trainer, LightningModule
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.optim import Optimizer
from wandb.plots.heatmap import heatmap

from tali.base import utils
from tali.sample_actuate_pred_and_data import (
    decode_and_store_text,
    make_image_frame_grid,
)

log = utils.get_logger(__name__)


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if trainer.fast_dev_run:
        raise Exception(
            "Cannot use wandb callbacks since pytorch lightning disables loggers in "
            "`fast_dev_run=true` mode."
        )

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for "
        "some reason..."
    )


class WatchModel(Callback):
    """Make wandb watch model at the beginning of the run."""

    def __init__(self, log: str = "gradients", log_freq: int = 100):
        self.log = log
        self.log_freq = log_freq

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        logger.watch(
            model=trainer.model,
            log=self.log,
            log_freq=self.log_freq,
            log_graph=False,
        )


class UploadCodeAsArtifact(Callback):
    """Upload all code files to wandb as an artifact, at the beginning of the run."""

    def __init__(self, code_dir: str):
        """

        Args:
            code_dir: the code directory
            use_git: if using git, then upload all files that are not ignored by git.
            if not using git, then upload all '*.py' file
        """
        self.code_dir = code_dir

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        code = wandb.Artifact("project-source", type="code")

        for path in Path(self.code_dir).resolve().rglob("*.py"):
            code.add_file(str(path), name=str(path.relative_to(self.code_dir)))

        experiment.log_artifact(code)


class UploadCheckpointsAsArtifact(Callback):
    """Upload checkpoints to wandb as an artifact, at the end of run."""

    def __init__(self, ckpt_dir: str = "checkpoints/", upload_best_only: bool = False):
        self.ckpt_dir = ckpt_dir
        self.upload_best_only = upload_best_only

    @rank_zero_only
    def on_keyboard_interrupt(self, trainer, pl_module):
        self.on_train_end(trainer, pl_module)

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in Path(self.ckpt_dir).rglob("*.ckpt"):
                ckpts.add_file(str(path))

        experiment.log_artifact(ckpts)


class LogConfusionMatrix(Callback):
    """Generate confusion matrix every epoch and send it to wandb.
    Expects validation step to return predictions and targets.
    """

    def __init__(self):
        self.preds = []
        self.targets = []
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module) -> None:
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        if self.ready:
            self.preds.append(outputs["preds"])
            self.targets.append(outputs["targets"])

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate confusion matrix."""
        if self.ready:
            logger = get_wandb_logger(trainer)
            experiment = logger.experiment

            preds = torch.cat(self.preds).cpu().numpy()
            targets = torch.cat(self.targets).cpu().numpy()

            confusion_matrix = metrics.confusion_matrix(y_true=targets, y_pred=preds)

            # set figure size
            plt.figure(figsize=(14, 8))

            # set labels size
            sn.set(font_scale=1.4)

            # set font size
            sn.heatmap(confusion_matrix, annot=True, annot_kws={"size": 8}, fmt="g")

            # names should be uniqe or else charts from different experiments in wandb will overlap
            experiment.log(
                {f"confusion_matrix/{experiment.name}": wandb.Image(plt)},
                commit=False,
            )

            # according to wandb docs this should also work but it crashes
            # experiment.log(f{"confusion_matrix/{experiment.name}": plt})

            # reset plot
            plt.clf()

            self.preds.clear()
            self.targets.clear()


class LogF1PrecRecHeatmap(Callback):
    """Generate f1, precision, recall heatmap every epoch and send it to wandb.
    Expects validation step to return predictions and targets.
    """

    def __init__(self, class_names: List[str] = None):
        self.preds = []
        self.targets = []
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        if self.ready:
            self.preds.append(outputs["preds"])
            self.targets.append(outputs["targets"])

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate f1, precision and recall heatmap."""
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment

            preds = torch.cat(self.preds).cpu().numpy()
            targets = torch.cat(self.targets).cpu().numpy()
            f1 = f1_score(targets, preds, average=None)
            r = recall_score(targets, preds, average=None)
            p = precision_score(targets, preds, average=None)
            data = [f1, p, r]

            # set figure size
            plt.figure(figsize=(14, 3))

            # set labels size
            sn.set(font_scale=1.2)

            # set font size
            sn.heatmap(
                data,
                annot=True,
                annot_kws={"size": 10},
                fmt=".3f",
                yticklabels=["F1", "Precision", "Recall"],
            )

            # names should be uniqe or else charts from different experiments in wandb will overlap
            experiment.log(
                {f"f1_p_r_heatmap/{experiment.name}": wandb.Image(plt)},
                commit=False,
            )

            # reset plot
            plt.clf()

            self.preds.clear()
            self.targets.clear()


def get_video_log_file(video_tensor: torch.Tensor) -> wandb.Video:
    video_tensor = video_tensor.cpu().permute([0, 2, 3, 1]).cpu().numpy()
    video_shape = video_tensor.shape
    video_tensor = video_tensor.reshape(
        -1, video_tensor.shape[2], video_tensor.shape[3]
    )
    video_tensor = cv2.cvtColor(video_tensor, cv2.COLOR_BGR2RGB)
    video_tensor = video_tensor.reshape(video_shape) * 255
    video_tensor = torch.Tensor(video_tensor).permute([0, 3, 1, 2]).type(torch.uint8)

    return wandb.Video(video_tensor, fps=8, format="gif")


def get_image_log_file(image_tensor: torch.Tensor) -> wandb.Image:
    return wandb.Image(
        make_image_frame_grid(
            image_frames=image_tensor.cpu().unsqueeze(0),
            num_video_frames_per_datapoint=1,
            save=False,
            store_dir=None,
            filename=None,
            show=False,
        )
    )


def get_audio_log_file(audio_tensor: torch.Tensor) -> wandb.Audio:
    return wandb.Audio(audio_tensor.cpu().permute([1, 0]), sample_rate=44100)


class LogMultiModalPredictionHeatmaps(Callback):
    """Logs a validation batch and their predictions to wandb.
    Example adapted from:
        https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY
    """

    def __init__(self, num_samples):
        super().__init__()
        self.ready = True
        self.num_samples = num_samples

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    @rank_zero_only
    def build_data_table(self, data_dict):
        available_modalities = [
            key for key, value in data_dict.items() if value is not None
        ]

        log_batch_dict = {
            "filepath": [
                filepath.replace(os.environ.get("DATASET_DIR"), "")
                .replace("full_video_360p", "")
                .replace(".frames", "")
                for filepath in data_dict["filepath"]
            ]
        }

        if "video" in available_modalities:
            log_batch_dict["video"] = [
                get_video_log_file(item) for item in data_dict["video"]
            ]

        if "image" in available_modalities:
            log_batch_dict["image"] = [
                get_image_log_file(item) for item in data_dict["image"]
            ]

        if "audio" in available_modalities:
            log_batch_dict["audio"] = [
                get_audio_log_file(item) for item in data_dict["audio"]
            ]

        if "text" in available_modalities:
            log_batch_dict["text"] = decode_and_store_text(
                text_frames=data_dict["text"].cpu(),
                save=False,
                show=False,
            )

        batch_zip = (list(value) for value in log_batch_dict.values())
        zip_list = list(batch_zip)
        rows = list(zip(*zip_list))
        return (
            wandb.Table(columns=list(log_batch_dict.keys()), data=rows),
            log_batch_dict["filepath"],
        )

    @rank_zero_only
    def log_similarity_heatmaps_multi_modal(self, trainer, pl_module, set_name):
        if not self.ready:
            return
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        # get a validation batch from the validation dat loader
        if set_name == "train":
            data_dict = next(iter(trainer.datamodule.train_dataloader()))
        elif set_name == "validation":
            data_dict = next(iter(trainer.datamodule.val_dataloader()))
        else:
            data_dict = next(iter(trainer.datamodule.test_dataloader()))

        data_dict = {key: value[: self.num_samples] for key, value in data_dict.items()}

        data_dict = {
            key: value.to(device=pl_module.device)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in data_dict.items()
        }

        (
            embedding_feature_dict,
            cross_modal_cosine_similarities,
            targets,
        ) = pl_module(data_dict)

        report_table, filepath_batch = self.build_data_table(data_dict)
        modalities = ["video", "image", "audio", "text"]
        output_similarities_collected = set()
        experiment.log(data={f"{set_name}-similarity-batch": report_table})

        for source_modality in modalities:
            for target_modality in modalities:

                if (
                    f"{source_modality}_{target_modality}_preds"
                    not in output_similarities_collected
                    and f"{source_modality}_to_{target_modality}_similarity"
                    in cross_modal_cosine_similarities
                ):
                    cur_preds = cross_modal_cosine_similarities[
                        f"{source_modality}_to_{target_modality}_similarity"
                    ]

                    cur_preds = (
                        cur_preds.clone()
                        .detach()
                        .cpu()
                        .requires_grad_(False)
                        .type(torch.float32)
                        .numpy()
                    )
                    cur_preds = np.around(cur_preds, decimals=3)

                    output_similarities_collected.add(
                        f"{source_modality}-{target_modality}-preds"
                    )

                    output_similarities_collected.add(
                        f"{target_modality}-{source_modality}-preds"
                    )

                    experiment.log(
                        {
                            f"{set_name}_heatmap"
                            f"_x={source_modality}"
                            f"_y={target_modality}": heatmap(
                                x_labels=[f"{filepath}" for filepath in filepath_batch],
                                y_labels=[f"{filepath}" for filepath in filepath_batch],
                                matrix_values=cur_preds,
                                show_text=True,
                            )
                        }
                    )

    @rank_zero_only
    def on_test_epoch_start(self, trainer, pl_module):
        self.log_similarity_heatmaps_multi_modal(trainer, pl_module, "test")

    @rank_zero_only
    def on_validation_epoch_start(self, trainer, pl_module):
        self.log_similarity_heatmaps_multi_modal(trainer, pl_module, "validation")

    @rank_zero_only
    def on_train_epoch_start(self, trainer, pl_module):
        self.log_similarity_heatmaps_multi_modal(trainer, pl_module, "train")

    @rank_zero_only
    def on_save_checkpoint(
        self,
        trainer: "pl.CustomTrainer",
        pl_module: "pl.LightningModule",
        checkpoint: Dict[str, Any],
    ) -> dict:
        self.log_similarity_heatmaps_multi_modal(trainer, pl_module, "train")


class LogGrads(Callback):
    """Logs a validation batch and their predictions to wandb.
    Example adapted from:
        https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY
    """

    def __init__(self, refresh_rate: int = 100):
        super().__init__()
        self.refresh_rate = refresh_rate

    @rank_zero_only
    def on_before_optimizer_step(
        self,
        trainer: "pl.CustomTrainer",
        pl_module: "pl.LightningModule",
        optimizer: Optimizer,
        opt_idx: int,
    ) -> None:

        if trainer.global_step % self.refresh_rate == 0:
            grad_dict = {
                name: param.grad.cpu().detach().abs().mean()
                for name, param in pl_module.named_parameters()
                if param.requires_grad and param.grad is not None
            }

            modality_keys = ["image", "video", "audio", "text"]

            modality_specific_grad_summary = {
                modality_key: [
                    value for key, value in grad_dict.items() if modality_key in key
                ]
                for modality_key in modality_keys
            }

            modality_specific_grad_summary = {
                key: {"x": np.arange(len(value)), "y": value}
                for key, value in modality_specific_grad_summary.items()
            }

            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment

            for key, value in modality_specific_grad_summary.items():
                data = [[x, y] for (x, y) in zip(value["x"], value["y"])]
                table = wandb.Table(data=data, columns=["x", "y"])
                experiment.log(
                    {
                        f"{key}_grad_summary": wandb.plot.line(
                            table,
                            "x",
                            "y",
                            title=f"{key} gradient summary",
                        )
                    }
                )


class LogConfigInformation(Callback):
    """Logs a validation batch and their predictions to wandb.
    Example adapted from:
        https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY
    """

    def __init__(self, config=None):
        super().__init__()
        self.done = False
        self.config = config

    @rank_zero_only
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self.done:
            logger = get_wandb_logger(trainer=trainer)

            trainer_hparams = trainer.__dict__.copy()

            hparams = {
                "trainer": trainer_hparams,
            }

            logger.log_hyperparams(hparams)
            logger.log_hyperparams(self.config)
            self.done = True
