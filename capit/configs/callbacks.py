from dataclasses import MISSING, dataclass
from datetime import timedelta
from typing import Dict, Optional

from capit.base.callbacks.wandb_callbacks import (LogConfigInformation, LogGrads,
                                                  SaveCheckpointsWandb,
                                                  UploadCodeAsArtifact)
from capit.base.utils.typing_utils import get_module_import_path
from capit.configs.string_variables import CHECKPOINT_DIR
from hydra_zen import builds, hydrated_dataclass
from pytorch_lightning.callbacks import (LearningRateMonitor, ModelCheckpoint,
                                         RichModelSummary, TQDMProgressBar)


@hydrated_dataclass(target=timedelta)
class TimerConfig:
    minutes: int = 60


@hydrated_dataclass(target=ModelCheckpoint)
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
    dirpath: str = CHECKPOINT_DIR


@hydrated_dataclass(target=RichModelSummary)
class ModelSummaryConfig:
    max_depth: int = 7


@hydrated_dataclass(target=TQDMProgressBar)
class RichProgressBar:
    refresh_rate: int = 1
    process_position: int = 0


@hydrated_dataclass(target=LearningRateMonitor)
class LearningRateMonitor:
    logging_interval: str = "step"


@hydrated_dataclass(target=UploadCodeAsArtifact)
class UploadCodeAsArtifact:
    code_dir: str = "${code_dir}"


@hydrated_dataclass(target=LogGrads)
class LogGrads:
    refresh_rate: int = 100


@hydrated_dataclass(target=LogConfigInformation)
class LogConfigInformation:
    config: Optional[Dict] = None


SaveCheckpointsWandb = builds(SaveCheckpointsWandb, populate_full_signature=True)

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
