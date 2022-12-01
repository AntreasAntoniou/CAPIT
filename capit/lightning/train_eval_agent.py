from typing import Any

import torch
from hydra_zen import instantiate
from pytorch_lightning import LightningDataModule, LightningModule

from capit.base.utils import get_logger, pretty_config

log = get_logger(__name__)


def get_dict_shapes(x):
    if not isinstance(x, dict):
        return get_dict_shapes(x.__dict__)
    return {
        key: value.shape if isinstance(value, torch.Tensor) else len(value)
        for key, value in x.items()
    }


class TrainingEvaluationAgent(LightningModule):
    def __init__(
        self,
        model_config: Any,
        optimizer_config: Any,
        datamodule: LightningDataModule,
    ):
        super().__init__()
        log.info(f"Initializing {self.__class__.__name__}")
        self.model: torch.nn.Module = instantiate(
            model_config,
            _recursive_=False,
        )
        self.optimizer_config = optimizer_config
        self.build(datamodule)

    def build(self, datamodule):
        batch = next(iter(datamodule.train_dataloader()))

        _, output_dict = self.model.build(batch)

        log.info(
            f"Built {self.__class__.__name__} "
            f"with {self.model.__class__.__name__} "
            f"and output shape "
            f"{pretty_config(get_dict_shapes(output_dict), resolve=True)} "
        )

    def forward(self, batch):
        return self.model.forward(batch, batch_idx=0)

    def training_step(self, batch, batch_idx, top_level_pl_module=None):
        opt_loss, output_dict = self.model.step(
            batch,
            batch_idx,
        )
        self.collect_metrics_step(output_dict["metrics"], phase_name="train")
        return opt_loss

    def validation_step(self, batch, batch_idx):
        _, output_dict = self.model.step(
            batch,
            batch_idx,
        )
        self.collect_metrics_step(output_dict["metrics"], phase_name="validation")

    def test_step(self, batch, batch_idx):
        _, output_dict = self.model.step(
            batch,
            batch_idx,
        )
        self.collect_metrics_step(output_dict["metrics"], phase_name="test")

    def configure_optimizers(self):
        optimizer = instantiate(
            config=self.optimizer_config,
            params=self.model.parameters(),
            lr=1e-6,
            _recursive_=False,
        )
        for key, value in self.named_parameters():
            log.info(
                f"Parameter {key} -> {value.shape} requires grad {value.requires_grad}"
            )
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer=optimizer,
        #     eta_min=float(self.optimizer_config.lr / 100),
        #     T_max=50000,
        #     verbose=True,
        # )
        # scheduler_dict = dict(
        #     scheduler=scheduler,
        #     interval="step",
        #     name="lr_scheduler",
        # )
        # print(
        #     scheduler_dict,
        #     dict(
        #         eta_min=float(self.optimizer_config.lr / 100),
        #         T_max=50000,
        #         verbose=True,
        #     ),
        # )
        return [optimizer]

    # , [scheduler_dict]

    def collect_metrics_step(self, metrics_dict, phase_name):
        for metric_key, computed_value in metrics_dict.items():

            if computed_value is not None:
                self.log(
                    name=f"{phase_name}/{metric_key}",
                    value=computed_value.detach(),
                    prog_bar=True if "opt_loss" in metric_key else False,
                    logger=True,
                    on_step=True,
                    on_epoch=True,
                    sync_dist=True,
                )
