from dataclasses import MISSING, dataclass, field
from typing import List

from torch.optim import AdamW

from capit.base.utils.typing_utils import get_module_import_path


@dataclass
class BaseOptimizerConfig:
    lr: float = MISSING
    _target_: str = MISSING


@dataclass
class AdamWOptimizerConfig(BaseOptimizerConfig):
    _target_: str = get_module_import_path(AdamW)
    lr: float = 2e-5
    weight_decay: float = 0.00001
    amsgrad: bool = False
    betas: List = field(default_factory=lambda: [0.9, 0.999])
