import os
from pathlib import Path
from typing import Any, Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import torch
import wandb
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.optim import Optimizer
from wandb.plots.heatmap import heatmap

from capit.base import utils

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


class PostBuildSummary(Callback):
    """
    Callback to log model summary after model is built
    """

    def __init__(self, max_depth):
        super(PostBuildSummary, self).__init__()
        self.max_depth = max_depth

    def on_sanity_check_end(
        self, trainer: "Trainer", pl_module: "LightningModule"
    ) -> None:
        summary = ModelSummary(model=pl_module, max_depth=self.max_depth)
        logging.info(summary)
