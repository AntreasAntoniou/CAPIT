import logging
from itertools import combinations
from typing import Any, Dict, List, Optional, Union

import hydra.utils
import numpy as np
import torch
from pytorch_lightning import LightningModule
from torchmetrics import Metric
from torchmetrics.classification.accuracy import Accuracy

from tali.base import utils
from tali.config_repository import (
    AutoAveragerConfig,
    AutoCLIPResNetConfig,
    AutoCLIPTextTransformerConfig,
    AutoCLIPVisionTransformerConfig,
    AutoConv1DTransformersConfig,
    AutoVideoTransformersConfig,
)
from tali.utils.device_utils import get_current_gpu_memory_stats

log = utils.get_logger(__name__)


def contrastive_logits_labels(logits: torch.Tensor):
    # logit shape is expected to be (batch_size, num_classes)
    labels = torch.arange(logits.shape[1]).type_as(logits).long()
    return logits, labels


class CrossModalMatchingNetwork(torch.nn.Module):
    def __init__(
        self,
        embedding_output_features: int,
        modality_embeddings: torch.nn.ModuleDict,
        logit_scale: float = 1 / 0.07,
        sub_batch_size_dict: Optional[Dict[str, int]] = None,
    ):
        super(CrossModalMatchingNetwork, self).__init__()
        log.info(f"Initializing {self.__class__.__name__}")
        log.info(f"{get_current_gpu_memory_stats()}")
        self.logit_scale_dict = None
        self.embed_dim = embedding_output_features
        self.modality_embeddings = modality_embeddings
        self.sub_batch_size_dict = sub_batch_size_dict
        self.logit_scale = np.log(logit_scale)
        self.is_built = False

    def init_logit_scale_params(self):
        modality_keys = [
            key for key, value in self.modality_embeddings.items() if value is not None
        ]

        for key, value in self.modality_embeddings.items():
            log.debug(f"{key} {value}")

        modality_combinations = combinations(modality_keys, 2)

        self.logit_scale_dict = {
            f"{modality_name[0]}_to_{modality_name[1]}": idx
            for idx, modality_name in enumerate(modality_combinations)
        }

        self.logit_scale_params = torch.nn.Parameter(
            torch.ones([len(self.logit_scale_dict)]) * self.logit_scale,
            requires_grad=True,
        )

        log.debug(f"Initialized logit scale params: {self.logit_scale_dict}")

    def build(
        self,
        batch_shape,
    ):

        logging.debug(f"{batch_shape}")

        for modality_key, modality_shape in batch_shape.items():
            log.info(
                f"Building embedding for {modality_key} processing {modality_shape} \n"
                f"current gpu device info: {get_current_gpu_memory_stats()}"
            )
            if modality_shape is not None:
                modality_shape = list(modality_shape)
                modality_shape[0] = 1
                if (
                    modality_key == "video"
                    and "image" in batch_shape
                    and batch_shape["image"] is not None
                ):
                    if hasattr(
                        self.modality_embeddings["video"], "connect_image_embedding"
                    ):
                        self.modality_embeddings["video"].connect_image_embedding(
                            self.modality_embeddings["image"]
                        )

                self.modality_embeddings[modality_key].build(modality_shape)
                self._check_modality_embedding_shape(modality_shape, modality_key)

        self.init_logit_scale_params()
        logging.debug(
            f"built {self.__class__.__name__} with output shape {self.embed_dim}",
        )
        log.info(f"Built system made of {self}")
        self.is_built = True

    def _check_modality_embedding_shape(self, input_shape, embedding_name):
        input_dummy = torch.zeros(size=input_shape)

        embeddings, _ = self.modality_embeddings[embedding_name].forward(input_dummy)

        assert embeddings.shape[1] == self.embed_dim

    def _get_normalized_features(self, inputs, embedding_name):

        if inputs is not None:
            inputs, _ = self.modality_embeddings[embedding_name](inputs)
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
            return inputs
        else:
            return None

    def _compute_cross_modal_cosine_similarities(self, embedding_dict):
        logit_dict = {}
        # log.info(
        #     f"Computing cross modal cosine similarities for {list(self.logit_scale_dict.keys())}"
        # )
        for source_key, source_value in embedding_dict.items():
            for target_key, target_value in embedding_dict.items():
                if (
                    source_key != target_key
                    and source_value is not None
                    and target_value is not None
                ):
                    if f"{source_key}_to_{target_key}" in self.logit_scale_dict:
                        logit_scale_idx = self.logit_scale_dict[
                            f"{source_key}_to_{target_key}"
                        ]
                    else:
                        logit_scale_idx = self.logit_scale_dict[
                            f"{target_key}_to_{source_key}"
                        ]

                    logit_scale = self.logit_scale_params[logit_scale_idx].exp()

                    if f"{target_key}_to_{source_key}_similarity" in logit_dict:
                        logit_dict[
                            f"{source_key}_to_{target_key}_similarity"
                        ] = logit_dict[f"{target_key}_to_{source_key}_similarity"].t()
                    else:
                        logit_dict[f"{source_key}_to_{target_key}_similarity"] = (
                            torch.matmul(source_value, target_value.t()) * logit_scale
                        )

        return logit_dict

    def forward(self, batch):

        batch = {
            key: value
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }

        if not self.is_built:
            log.info(
                f"Building during forward in "
                f"{self.__class__.__name__} "
                f"{get_current_gpu_memory_stats()}"
            )
            self.build({key: value.shape for key, value in batch.items()})

        if self.sub_batch_size_dict is not None:
            embedding_feature_dict = {}
            for embedding_name, inputs in batch.items():
                if embedding_name in self.sub_batch_size_dict:
                    sub_batch_size = self.sub_batch_size_dict[embedding_name]
                    sub_batches = inputs.view(
                        (
                            sub_batch_size,
                            -1,
                        )
                        + inputs.shape[1:]
                    )
                    current_embedding_features = []
                    for sub_batch in sub_batches:
                        sub_batch_features_for_current_embedding = (
                            self._get_normalized_features(sub_batch, embedding_name)
                        )
                        current_embedding_features.append(
                            sub_batch_features_for_current_embedding
                        )
                    embedding_feature_dict[embedding_name] = torch.cat(
                        current_embedding_features, dim=0
                    )
                else:
                    embedding_feature_dict[
                        embedding_name
                    ] = self._get_normalized_features(inputs, embedding_name)
        else:
            embedding_feature_dict = {
                embedding_name: self._get_normalized_features(inputs, embedding_name)
                for embedding_name, inputs in batch.items()
            }

        return (
            embedding_feature_dict,
            self._compute_cross_modal_cosine_similarities(embedding_feature_dict),
        )


class CrossEntropyLoss(Metric):
    higher_is_better = False

    def __init__(
        self,
        dist_sync_on_step: bool = False,
    ) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("loss_sum", default=torch.zeros([]), dist_reduce_fx="sum")
        self.add_state("num_updates", default=torch.zeros([]), dist_reduce_fx="sum")

        self.criterion = torch.nn.CrossEntropyLoss()

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # update metric states
        self.loss_sum += self.criterion(preds.float(), target.long()).item()
        self.num_updates += 1

    def compute(self):
        # compute metric

        loss = self.loss_sum / self.num_updates
        return loss


class ModusPrime(LightningModule):
    def __init__(
        self,
        name: str,
        image_embedding_config: Union[
            AutoCLIPVisionTransformerConfig, AutoCLIPResNetConfig
        ],
        audio_embedding_config: Union[AutoConv1DTransformersConfig],
        text_embedding_config: Union[AutoCLIPTextTransformerConfig],
        video_embedding_config: Union[AutoAveragerConfig, AutoVideoTransformersConfig],
        optimizer_config: None,
        lr_scheduler_config: None,
        sub_batch_size_dict: Optional[Dict[str, int]] = None,
        batch_size: int = 2,
        num_train_samples: int = None,
        embedding_output_features: int = 512,
        logit_scale: float = 1 / 0.07,
    ):
        super(ModusPrime, self).__init__()

        log.info(f"Initializing {self.__class__.__name__}")
        log.info(f"{get_current_gpu_memory_stats()}")

        modality_embeddings = torch.nn.ModuleDict()
        log.info(f"Instantiating image embedding <{image_embedding_config._target_}>")
        modality_embeddings["image"] = hydra.utils.instantiate(image_embedding_config)

        log.info(f"Instantiating audio embedding <{audio_embedding_config._target_}>")
        modality_embeddings["audio"] = hydra.utils.instantiate(audio_embedding_config)

        log.info(f"Instantiating text embedding <{text_embedding_config._target_}>")
        modality_embeddings["text"] = hydra.utils.instantiate(text_embedding_config)

        log.info(f"Instantiating video embedding <{video_embedding_config._target_}>")
        modality_embeddings["video"] = hydra.utils.instantiate(video_embedding_config)

        self.system = CrossModalMatchingNetwork(
            embedding_output_features=embedding_output_features,
            modality_embeddings=modality_embeddings,
            logit_scale=logit_scale,
            sub_batch_size_dict=sub_batch_size_dict,
        )
        self.sub_batch_size_dict = sub_batch_size_dict
        self.batch_size = batch_size
        self.num_train_samples = num_train_samples
        self.is_built = False
        self.sync_dist = False

        self.metrics_to_track = {
            "cross_entropy": CrossEntropyLoss,
            "accuracy": Accuracy,
        }

        self.per_modality_metrics_computed_dict = torch.nn.ModuleDict(
            {
                "training-metrics": torch.nn.ModuleDict(),
                "validation-metrics": torch.nn.ModuleDict(),
                "test-metrics": torch.nn.ModuleDict(),
            }
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer_config = optimizer_config
        self.lr_scheduler_config = lr_scheduler_config
        self.save_hyperparameters(logger=True)

    def build(self, batch):
        (embedding_feature_dict, logits_similarities_dict,) = self.system.forward(
            batch,
        )

        targets_dict = {
            key: contrastive_logits_labels(value)[1]
            for key, value in logits_similarities_dict.items()
        }

        for phase_name in ["training", "validation", "test"]:
            self.init_metrics(
                logits_dict=logits_similarities_dict,
                targets_dict=targets_dict,
                phase_name=phase_name,
            )

        self.is_built = True

    def reset_metric_caches(self, phase_name):
        for key in self.per_modality_metrics_computed_dict[
            f"{phase_name}-metrics"
        ].keys():
            self.per_modality_metrics_computed_dict[f"{phase_name}-metrics"][
                key
            ].reset()

    def forward(self, batch):

        if not self.is_built:
            self.build(
                batch=batch,
            )

        (embedding_feature_dict, logits_similarities_dict,) = self.system.forward(
            batch,
        )

        targets_dict = {
            key: contrastive_logits_labels(value)[1]
            for key, value in logits_similarities_dict.items()
        }

        return embedding_feature_dict, logits_similarities_dict, targets_dict

    def init_metrics(self, logits_dict, targets_dict, phase_name):
        for metric_key, metric_function in self.metrics_to_track.items():
            for measurement_key, measurement_value, target_value in zip(
                logits_dict.keys(), logits_dict.values(), targets_dict.values()
            ):
                # logging.debug(f"{measurement_value'].shape} {target_value.shape}")
                cur_key = f"{metric_key}_{measurement_key}"

                if (
                    cur_key
                    not in self.per_modality_metrics_computed_dict[
                        f"{phase_name}-metrics"
                    ]
                ):
                    self.per_modality_metrics_computed_dict[f"{phase_name}-metrics"][
                        cur_key
                    ] = metric_function(dist_sync_on_step=self.sync_dist)
                    self.per_modality_metrics_computed_dict.to(self.device)
                    # log.info(list(self.per_modality_metrics_computed_dict.items()))

            cur_key = f"overall_{metric_key}"

            if (
                cur_key
                not in self.per_modality_metrics_computed_dict[f"{phase_name}-metrics"]
            ):
                self.per_modality_metrics_computed_dict[f"{phase_name}-metrics"][
                    cur_key
                ] = metric_function(dist_sync_on_step=self.sync_dist)
                self.per_modality_metrics_computed_dict.to(self.device)
                # log.info(list(self.per_modality_metrics_computed_dict.items()))
        log.info(list(self.per_modality_metrics_computed_dict.items()))

    def collect_metrics_step(self, logits_dict, targets_dict, phase_name):
        for key, value in self.system.logit_scale_dict.items():
            self.log(
                name=f"logit_scale/{key}",
                value=self.system.logit_scale_params[value].exp(),
                prog_bar=False,
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=self.sync_dist,
            )

        for metric_key, metric_function in self.metrics_to_track.items():
            for measurement_key, measurement_value, target_value in zip(
                logits_dict.keys(), logits_dict.values(), targets_dict.values()
            ):
                # logging.debug(f"{measurement_value'].shape} {target_value.shape}")
                cur_key = f"{metric_key}_{measurement_key}"

                value = self.per_modality_metrics_computed_dict[
                    f"{phase_name}-metrics"
                ][cur_key](
                    measurement_value.detach(),
                    target_value.detach(),
                )
                if value is not None:
                    self.log(
                        name=f"{phase_name}/{cur_key}",
                        value=value,
                        prog_bar=False,
                        logger=True,
                        on_step=True,
                        on_epoch=False,
                        sync_dist=self.sync_dist,
                    )

            cur_key = f"overall_{metric_key}"

            value = self.per_modality_metrics_computed_dict[f"{phase_name}-metrics"][
                cur_key
            ](
                torch.stack(list(logits_dict.values())).detach(),
                torch.stack(list(targets_dict.values())).detach(),
            )

            if value is not None:
                self.log(
                    name=f"{phase_name}/{cur_key}",
                    value=value,
                    prog_bar=False,
                    logger=True,
                    on_step=True,
                    on_epoch=True,
                    sync_dist=self.sync_dist,
                )

    def collect_metrics_epoch(self, phase_name):
        self.per_modality_metrics_computed_dict.to(self.device)
        for key, value in self.per_modality_metrics_computed_dict[
            f"{phase_name}-metrics"
        ].items():

            if isinstance(value, Accuracy) and value is not None:
                self.log(
                    name=f"{phase_name}/{key}/epoch",
                    value=value.compute().detach(),
                    prog_bar=False,
                    logger=True,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

            if isinstance(value, CrossEntropyLoss) and value is not None:
                self.log(
                    name=f"{phase_name}/{key}/epoch",
                    value=value.compute().detach(),
                    prog_bar=False,
                    logger=True,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

    def step(self, batch, batch_idx):

        return self.forward(batch)

    def training_step(self, batch, batch_idx):
        (
            embedding_feature_dict,
            logits_similarities_dict,
            targets_dict,
        ) = self.step(batch=batch, batch_idx=batch_idx)

        logits = torch.stack(list(logits_similarities_dict.values()), dim=0)
        logits = logits.view(-1, logits.shape[-1])

        targets = torch.stack(list(targets_dict.values()), dim=0)
        targets = targets.view(-1)

        loss = self.criterion(input=logits, target=targets)

        self.collect_metrics_step(
            logits_dict=logits_similarities_dict,
            targets_dict=targets_dict,
            phase_name="training",
        )

        if self.lr_scheduler_step_must_be_called_manually:
            self.lr_scheduler.step(loss.detach().item(), self.global_step)

        return loss

    def training_epoch_end(self, outputs: List[Any]):
        log.info(f"\nTraining epoch {self.current_epoch} ended.\n")
        self.collect_metrics_epoch(phase_name="training")
        self.reset_metric_caches(phase_name="training")

    def validation_step(self, batch, batch_idx):
        # logging.debug(f'{[(key, value.shape) for key, value in batch.items()]}')

        (
            embedding_feature_dict,
            logits_similarities_dict,
            targets_dict,
        ) = self.step(batch=batch, batch_idx=batch_idx)

        self.collect_metrics_step(
            logits_dict=logits_similarities_dict,
            targets_dict=targets_dict,
            phase_name="validation",
        )

    def validation_epoch_end(self, outputs: List[Any]):
        log.info(f"\nValidation epoch {self.current_epoch} ended.\n")
        self.collect_metrics_epoch(phase_name="validation")
        self.reset_metric_caches(phase_name="validation")

    def test_step(self, batch, batch_idx):
        # logging.debug(f'{[(key, value.shape) for key, value in batch.items()]}')

        (
            embedding_feature_dict,
            logits_similarities_dict,
            targets_dict,
        ) = self.step(batch=batch, batch_idx=batch_idx)

        self.collect_metrics_step(
            logits_dict=logits_similarities_dict,
            targets_dict=targets_dict,
            phase_name="test",
        )

    def test_epoch_end(self, outputs: List[Any]):
        log.info(f"\nTest epoch {self.current_epoch} ended.\n")
        self.collect_metrics_epoch(phase_name="test")
        self.reset_metric_caches(phase_name="test")

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            config=self.optimizer_config, params=self.parameters()
        )

        optimizer_dict = {"optimizer": optimizer}
        if self.lr_scheduler_config._target_.split(".")[-1] == "CosineAnnealingLR":
            if "T_max" not in self.lr_scheduler_config:
                self.lr_scheduler_config["T_max"] = (
                    self.num_train_samples / self.batch_size
                )
        elif (
            self.lr_scheduler_config._target_.split(".")[-1]
            == "CosineAnnealingWarmRestarts"
        ):
            if "T_0" not in self.lr_scheduler_config:
                self.lr_scheduler_config["T_0"] = (
                    self.num_train_samples / self.batch_size // 2
                )

        elif self.lr_scheduler_config._target_.split(".")[-1] == "ReduceLROnPlateau":
            self.lr_scheduler_config["patience"] = (
                self.lr_scheduler_config["patience"] * torch.cuda.device_count()
                if torch.cuda.is_available()
                else 1
            )

        lr_scheduler = hydra.utils.instantiate(
            config=self.lr_scheduler_config, optimizer=optimizer
        )
        log.info(
            f"\noptimizer: {optimizer} \n" f"lr_scheduler: {self.lr_scheduler_config}"
        )
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_scheduler = lr_scheduler
            self.lr_scheduler_step_must_be_called_manually = True
        else:
            self.lr_scheduler_step_must_be_called_manually = False
            optimizer_dict["lr_scheduler"] = {
                "scheduler": lr_scheduler,
                "interval": "step",
            }

        return optimizer_dict


class DumbusPrime(LightningModule):
    def __init__(
        self,
        name: str,
        image_embedding_config: Union[
            AutoCLIPVisionTransformerConfig, AutoCLIPResNetConfig
        ],
        audio_embedding_config: Union[AutoConv1DTransformersConfig],
        text_embedding_config: Union[AutoCLIPTextTransformerConfig],
        video_embedding_config: Union[AutoAveragerConfig, AutoVideoTransformersConfig],
        optimizer_config: None,
        lr_scheduler_config: None,
        sub_batch_size_dict: Optional[Dict[str, int]] = None,
        batch_size: int = 2,
        embedding_output_features: int = 512,
        logit_scale: float = 1.0 / 0.07,
    ):
        super(DumbusPrime, self).__init__()
        self.system = CrossModalMatchingNetwork(
            embedding_output_features=embedding_output_features,
            modality_embeddings=torch.nn.ModuleDict(
                dict(
                    image=hydra.utils.instantiate(image_embedding_config),
                    audio=hydra.utils.instantiate(audio_embedding_config),
                    text=hydra.utils.instantiate(text_embedding_config),
                    video=hydra.utils.instantiate(video_embedding_config),
                )
            ),
            logit_scale=logit_scale,
            sub_batch_size_dict=sub_batch_size_dict,
        )
        self.sub_batch_size_dict = sub_batch_size_dict
        self.batch_size = batch_size
        self.is_built = False

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer_config = optimizer_config
        self.lr_scheduler_config = lr_scheduler_config
        # self.save_hyperparameters(logger=False)

    def build(self, batch):
        _, _ = self.system.forward(batch=batch)

        self.is_built = True
        # self.save_hyperparameters(logger=False)

    def forward(self, batch):

        if not self.system.is_built:
            self.system.build(
                batch_shape={
                    key: value.shape if isinstance(value, torch.Tensor) else None
                    for key, value in batch.items()
                }
            )

        (
            embedding_feature_dict,
            cross_modal_cosine_similarities,
        ) = self.system.forward(
            batch,
        )

        targets = torch.stack(
            [
                contrastive_logits_labels(modality_similarities)[1]
                for modality_similarities in cross_modal_cosine_similarities.values()
            ],
            dim=0,
        )

        return embedding_feature_dict, cross_modal_cosine_similarities, targets

    def step(self, batch, batch_idx):

        (embedding_feature_dict, logits_similarities_dict,) = self.system.forward(
            batch,
        )

        logits_shape_dict = {
            key: value.shape for key, value in logits_similarities_dict.items()
        }

        log.info(f"logits_shape_dict: {logits_shape_dict}")

        targets_dict = {
            key: contrastive_logits_labels(value)[1]
            for key, value in logits_similarities_dict.items()
        }

        return embedding_feature_dict, logits_similarities_dict, targets_dict

    def training_step(self, batch, batch_idx):
        (
            embedding_feature_dict,
            logits_similarities_dict,
            targets_dict,
        ) = self.step(batch=batch, batch_idx=batch_idx)

        logits = torch.stack(list(logits_similarities_dict.values()), dim=0)
        logits = logits.view(-1, logits.shape[-1])

        targets = torch.stack(list(targets_dict.values()), dim=0)
        targets = targets.view(-1)

        loss = self.criterion(input=logits, target=targets)

        if self.lr_scheduler_step_must_be_called_manually:
            self.lr_scheduler.step(loss.detach().item(), self.global_step)

        return loss

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            config=self.optimizer_config, params=self.parameters()
        )

        optimizer_dict = {"optimizer": optimizer}
        if self.lr_scheduler_config._target_.split(".")[-1] == "CosineAnnealingLR":
            if "T_max" not in self.lr_scheduler_config:
                self.lr_scheduler_config["T_max"] = 1000000
        elif (
            self.lr_scheduler_config._target_.split(".")[-1]
            == "CosineAnnealingWarmRestarts"
        ):
            if "T_0" not in self.lr_scheduler_config:
                self.lr_scheduler_config["T_0"] = 1000000

        elif self.lr_scheduler_config._target_.split(".")[-1] == "ReduceLROnPlateau":
            self.lr_scheduler_config["patience"] = (
                self.lr_scheduler_config["patience"] * torch.cuda.device_count()
                if torch.cuda.is_available()
                else 1
            )

        lr_scheduler = hydra.utils.instantiate(
            config=self.lr_scheduler_config, optimizer=optimizer
        )
        log.info(
            f"\noptimizer: {optimizer} \n" f"lr_scheduler: {self.lr_scheduler_config}"
        )
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_scheduler = lr_scheduler
            self.lr_scheduler_step_must_be_called_manually = True
        else:
            self.lr_scheduler_step_must_be_called_manually = False
            optimizer_dict["lr_scheduler"] = {
                "scheduler": lr_scheduler,
                "interval": "step",
            }

        return optimizer_dict
