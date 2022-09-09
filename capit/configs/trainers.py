from dataclasses import dataclass
from typing import Any

from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPPlugin

from capit.base.utils.typing_utils import get_module_import_path
from capit.configs.string_variables import CURRENT_EXPERIMENT_DIR


@dataclass
class BaseTrainer:
    _target_: str = get_module_import_path(Trainer)
    gpus: int = 0
    accelerator: str = "cpu"
    enable_checkpointing: bool = True
    default_root_dir: str = CURRENT_EXPERIMENT_DIR
    enable_progress_bar: bool = True
    val_check_interval: float = 0.02
    max_steps: int = 10000
    log_every_n_steps: int = 1
    precision: int = 32
    num_sanity_val_steps: int = 2
    auto_scale_batch_size: bool = False


@dataclass
class DDPPlugin:
    _target_: str = get_module_import_path(DDPPlugin)
    find_unused_parameters: bool = False


@dataclass
class DDPTrainer(BaseTrainer):
    accelerator: str = "gpu"
    strategy: Any = None
    replace_sampler_ddp: bool = True
    sync_batchnorm: bool = True
    auto_scale_batch_size: bool = False
    plugins: Any = DDPPlugin()


@dataclass
class DPTrainer(BaseTrainer):
    accelerator: str = "gpu"
    strategy: str = "dp"
    auto_scale_batch_size: bool = False


@dataclass
class MPSTrainer(BaseTrainer):
    accelerator: str = "mps"
    gpus: int = 0
