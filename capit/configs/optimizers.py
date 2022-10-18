from dataclasses import MISSING, dataclass, field
from typing import List

from capit.base.utils.typing_utils import get_module_import_path
from hydra_zen import hydrated_dataclass
from torch.optim import AdamW


@dataclass
class BaseOptimizerConfig:
    lr: float = MISSING
    _target_: str = MISSING


@hydrated_dataclass(target=AdamW)
class AdamWOptimizerConfig(BaseOptimizerConfig):
    lr: float = 2e-5
    weight_decay: float = 0.00001
    amsgrad: bool = False
    betas: List = field(default_factory=lambda: [0.9, 0.999])
