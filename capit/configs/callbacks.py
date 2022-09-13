from dataclasses import MISSING, dataclass
from datetime import timedelta
from typing import Dict, Optional

from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    TQDMProgressBar,
)

from capit.base.callbacks.wandb_callbacks import (
    LogConfigInformation,
    LogGrads,
    UploadCodeAsArtifact,
)
from capit.base.utils.typing_utils import get_module_import_path
from capit.configs.string_variables import CHECKPOINT_DIR


@dataclass
class TimerConfig:
    _target_: str = get_module_import_path(timedelta)
    minutes: int = 15


@dataclass
class ModelCheckpointingConfig:
    monitor: str = MISSING
    mode: str = MISSING
    save_top_k: int = MISSING
    save_last: bool = MISSING
    verbose: bool = MISSING
    filename: str = MISSING
    auto_insert_metric_name: bool = MISSING
    save_on_train_epoch_end: Optional[bool] = None
    train_time_interval: Optional[TimerConfig] = None
    _target_: str = get_module_import_path(ModelCheckpoint)
    dirpath: str = CHECKPOINT_DIR


@dataclass
class ModelSummaryConfig:
    _target_: str = get_module_import_path(RichModelSummary)
    max_depth: int = 7


@dataclass
class RichProgressBar:
    _target_: str = get_module_import_path(TQDMProgressBar)
    refresh_rate: int = 1
    process_position: int = 0


@dataclass
class LearningRateMonitor:
    _target_: str = get_module_import_path(LearningRateMonitor)
    logging_interval: str = "step"


@dataclass
class UploadCodeAsArtifact:
    _target_: str = get_module_import_path(UploadCodeAsArtifact)
    code_dir: str = "${code_dir}"


@dataclass
class LogGrads:
    _target_: str = get_module_import_path(LogGrads)
    refresh_rate: int = 100


@dataclass
class LogConfigInformation:
    _target_: str = get_module_import_path(LogConfigInformation)
    config: Optional[Dict] = None


model_checkpoint_eval: ModelCheckpointingConfig = ModelCheckpointingConfig(
    monitor="validation/accuracy_epoch",
    mode="max",
    save_top_k=3,
    save_last=False,
    verbose=False,
    dirpath=CHECKPOINT_DIR,
    filename="eval_epoch",
    auto_insert_metric_name=False,
)

model_checkpoint_train = ModelCheckpointingConfig(
    _target_=get_module_import_path(ModelCheckpoint),
    monitor="training/loss_epoch",
    save_on_train_epoch_end=True,
    save_top_k=0,
    save_last=True,
    train_time_interval=TimerConfig(),
    mode="min",
    verbose=False,
    dirpath=CHECKPOINT_DIR,
    filename="train_latest",
    auto_insert_metric_name=False,
)
