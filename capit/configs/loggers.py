# https://wandb.ai
import os
from dataclasses import dataclass, field
from typing import List, Optional

from capit.base.utils.typing_utils import get_module_import_path
from capit.configs.string_variables import CURRENT_EXPERIMENT_DIR, EXPERIMENT_NAME
from hydra_zen import hydrated_dataclass
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


@hydrated_dataclass(target=WandbLogger)
class WeightsAndBiasesLoggerConfig:
    project: str = os.environ["WANDB_PROJECT"]
    offline: bool = False  # set True to store all logs only locally
    resume: str = "allow"  # allow, True, False, must
    save_dir: str = CURRENT_EXPERIMENT_DIR
    log_model: Optional[str] = "all"
    prefix: str = ""
    job_type: str = "train"
    group: str = ""
    tags: List[str] = field(default_factory=list)


@hydrated_dataclass(target=TensorBoardLogger)
class TensorboardLoggerConfig:
    save_dir: str = CURRENT_EXPERIMENT_DIR
    name: str = EXPERIMENT_NAME
    version: Optional[str] = None
    log_graph: bool = False
    default_hp_metric: Optional[bool] = None
