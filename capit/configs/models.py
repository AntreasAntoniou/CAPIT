from dataclasses import dataclass
from typing import Any

from capit.base.utils.typing_utils import get_module_import_path
from capit.models.image_text_models import CLIPImageTextModel


@dataclass
class CLIPImageTextMultiModalDatasetConfig:
    _target_: Any = get_module_import_path(CLIPImageTextModel)
    model_name_or_path: str = "openai/clip-vit-base-patch16"
