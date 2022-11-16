from dataclasses import dataclass
from typing import Any, Union

from capit.base.utils.typing_utils import get_module_import_path
from capit.configs.string_variables import CURRENT_EXPERIMENT_DIR
from hydra_zen import hydrated_dataclass
from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPPlugin


@hydrated_dataclass(target=Trainer)
class BaseTrainer:
    gpus: int = 0
    accelerator: str = "cpu"
    enable_checkpointing: bool = True
    default_root_dir: str = CURRENT_EXPERIMENT_DIR
    enable_progress_bar: bool = True
    val_check_interval: Union[int, float] = 0.001
    max_steps: int = 1000000
    log_every_n_steps: int = 1
    precision: int = 32
    num_sanity_val_steps: int = 2
    auto_scale_batch_size: bool = False


@hydrated_dataclass(target=DDPPlugin)
class DDPPlugin:
    find_unused_parameters: bool = False


@hydrated_dataclass(target=Trainer)
class DDPTrainer(BaseTrainer):
    accelerator: str = "gpu"
    strategy: Any = None
    replace_sampler_ddp: bool = True
    sync_batchnorm: bool = True
    auto_scale_batch_size: bool = False
    plugins: Any = DDPPlugin()


@hydrated_dataclass(target=Trainer)
class DPTrainer(BaseTrainer):
    accelerator: str = "gpu"
    strategy: str = "dp"
    auto_scale_batch_size: bool = False


@hydrated_dataclass(target=Trainer)
class MPSTrainer(BaseTrainer):
    accelerator: str = "mps"
    gpus: int = 0
