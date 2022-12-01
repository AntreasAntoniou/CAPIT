from dataclasses import dataclass
from typing import Any

from hydra_zen import hydrated_dataclass

from capit.base.utils.typing_utils import get_module_import_path
from capit.models.image_text_models import (
    CLIPImageTextModel,
    CLIPWithPostProcessingImageTextModel,
)


@dataclass
class ModelNames:
    clip_vit_large_patch14 = "openai/clip-vit-large-patch14"
    clip_vit_base_patch16 = "openai/clip-vit-base-patch16"


@hydrated_dataclass(target=CLIPImageTextModel)
class CLIPImageTextMultiModalDatasetConfig:
    model_name_or_path: str = ModelNames.clip_vit_large_patch14
    pretrained: bool = True


@hydrated_dataclass(target=CLIPWithPostProcessingImageTextModel)
class CLIPWithPostProcessingImageTextModelConfig:
    model_name_or_path: str = ModelNames.clip_vit_large_patch14
    pretrained: bool = True
    fine_tunable: bool = False
