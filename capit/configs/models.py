from dataclasses import dataclass
from typing import Any

from capit.base.utils.typing_utils import get_module_import_path
from capit.models.image_text_models import CLIPImageTextModel
from hydra_zen import hydrated_dataclass


@hydrated_dataclass(target=CLIPImageTextModel)
class CLIPImageTextMultiModalDatasetConfig:
    model_name_or_path: str = "openai/clip-vit-base-patch16"
    pretrained: bool = True
    fine_tunable: bool = False
