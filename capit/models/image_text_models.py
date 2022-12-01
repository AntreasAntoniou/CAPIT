from dataclasses import dataclass
from typing import Any, Iterator, Optional, Tuple
from unittest.util import strclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotted_dict import DottedDict
from matplotlib import image
from rich import print
from transformers import CLIPModel, CLIPProcessor
from transformers.models.clip.modeling_clip import CLIPOutput, contrastive_loss

from capit.base.utils import get_logger
from capit.decorators import hydra_configurable
from capit.models.helpers import contrastive_logits_labels

log = get_logger(__name__)


@dataclass
class CLIPModelOutput:
    logits_per_image: torch.Tensor
    text_embeds: torch.Tensor
    image_embeds: torch.Tensor
    loss: Optional[torch.Tensor] = None


@hydra_configurable
class CLIPImageTextModel(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        pretrained: bool = True,
    ):
        super().__init__()
        self.model: CLIPModel = CLIPModel.from_pretrained(model_name_or_path)
        self.processor: CLIPProcessor = CLIPProcessor.from_pretrained(
            model_name_or_path
        )

        self.pretrained = pretrained

        if not pretrained:
            self.model.init_weights()

        self.model.train()

        self.image_shape = [
            3,
            self.processor.feature_extractor.size,
            self.processor.feature_extractor.size,
        ]
        self.is_build = False

    def build(self, batch):
        log.info(f"Built model {self.__class__.__name__}")
        image = batch["target_image"][0]
        challenge_images = batch["challenge_images"][0]
        image = torch.cat([image.unsqueeze(0), challenge_images], dim=0)
        text = batch["target_text"][0]

        image = self.preprocess_image(image)
        text = self.preprocess_text(text)

        if len(text.shape) == 1:
            text = text.unsqueeze(0)

        clip_output = self.model.forward(
            input_ids=text,
            pixel_values=image,
            output_hidden_states=True,
            return_loss=False,
        )

        image_hidden_token = clip_output.vision_model_output.hidden_states[-1]
        text_hidden_token = clip_output.text_model_output.hidden_states[-1]

        self.is_built = True

        return self.step(batch, batch_idx=0)

    def preprocess_image(self, image: torch.Tensor):
        image = self.processor(images=image, return_tensors="pt")["pixel_values"]
        image = image.to(self.model.device)

        if len(image.shape) != 4:
            raise ValueError(
                f"Input shape for class {self.__class__.__name__} in "
                f"method forward_image must be 4, instead it is "
                f"{len(image.shape)}, for shape {image.shape}"
            )
        return image

    def preprocess_text(self, text: torch.Tensor) -> torch.Tensor:
        text = self.processor(
            text=text, return_tensors="pt", padding=True, truncation=True
        )["input_ids"]
        text = text.to(self.model.device)
        text = text.to(torch.int32)
        return text

    def forward_image(self, image: torch.Tensor) -> torch.Tensor:
        image = self.preprocess_image(image)
        clip_output = self.model.forward(pixel_values=image)

        image_hidden_token = clip_output.vision_model_output.hidden_states[-1]

        return image_hidden_token

    def forward_text(self, text: torch.Tensor) -> torch.Tensor:

        text = self.preprocess_text(text)
        if len(text.shape) == 1:
            text = text.unsqueeze(0)
        clip_output = self.model.forward(input_ids=text)

        text_hidden_token = clip_output.vision_model_output.hidden_states[-1]

        return text_hidden_token

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> CLIPOutput:

        image = self.preprocess_image(image)
        text = self.preprocess_text(text)

        if len(text.shape) == 1:
            text = text.unsqueeze(0)

        clip_output = self.model.forward(
            input_ids=text,
            pixel_values=image,
            output_hidden_states=True,
            return_loss=False,
        )

        image_hidden_token = clip_output.image_embeds
        text_hidden_token = clip_output.text_embeds

        similarity = (
            torch.matmul(text_hidden_token, image_hidden_token.t())
            * self.model.logit_scale
        )

        loss = contrastive_loss(similarity)

        return CLIPModelOutput(
            logits_per_image=similarity,
            image_embeds=image_hidden_token,
            text_embeds=text_hidden_token,
            loss=loss,
        )

    def predict_individual(
        self, image: torch.Tensor, text: torch.Tensor
    ) -> CLIPModelOutput:
        image_embeds = self.forward_image(image)
        text_embeds = self.forward_text(text)

        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits

        logit_scale = self.model.logit_scale.exp()
        return torch.sum(text_embeds * image_embeds, dim=1) * logit_scale

    def step(self, batch, batch_idx):

        image = batch["target_image"][0]
        challenge_images = batch["challenge_images"][0]
        images = torch.cat([image.unsqueeze(0), challenge_images], dim=0)
        text = batch["target_text"][0]

        clip_output = self.forward(image=images, text=text)

        accuracy = (clip_output.logits_per_image.argmax(dim=-1) == 0).float().mean()
        output_dict = clip_output.__dict__
        output_dict["metrics"] = {"accuracy": accuracy, "loss": clip_output.loss}

        return output_dict["loss"], output_dict


@hydra_configurable
class CLIPWithPostProcessingImageTextModel(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        pretrained: bool = True,
        fine_tunable: bool = True,
    ):
        super().__init__()
        self.model: CLIPModel = CLIPModel.from_pretrained(model_name_or_path)
        self.processor: CLIPProcessor = CLIPProcessor.from_pretrained(
            model_name_or_path
        )

        self.pretrained = pretrained
        self.fine_tunable = fine_tunable

        if not pretrained:
            self.model.init_weights()

        self.model.train()

        self.image_shape = [
            3,
            self.processor.feature_extractor.size,
            self.processor.feature_extractor.size,
        ]
        self.is_build = False
        self.post_processing_module = nn.ModuleDict()

    def parameters(self):
        if self.fine_tunable:
            return list(self.model.parameters()) + list(
                self.post_processing_module.parameters()
            )
        else:
            return self.post_processing_module.parameters()

    def named_parameters(self) -> Iterator[Tuple[str, torch.Tensor]]:
        if self.fine_tunable:
            return list(self.model.named_parameters()) + list(
                self.post_processing_module.named_parameters()
            )

        else:
            return self.post_processing_module.named_parameters()

    def build_post_processing_module(self, name: str, x: torch.Tensor):
        transformer_encoder = nn.TransformerEncoderLayer(
            d_model=x.shape[2], nhead=8, dim_feedforward=2048
        )
        encoder_norm = nn.LayerNorm(x.shape[2])
        self.post_processing_module[f"{name}_transformer"] = nn.TransformerEncoder(
            encoder_layer=transformer_encoder, num_layers=1, norm=encoder_norm
        )
        x = self.post_processing_module[f"{name}_transformer"](x)
        x = x.mean(dim=1)
        self.post_processing_module[f"{name}_output"] = nn.Linear(x.shape[1], 512)
        x = self.post_processing_module[f"{name}_output"](x)
        return x

    def apply_post_processing(self, x: torch.Tensor, name: str):
        x = self.post_processing_module[f"{name}_transformer"](x)
        x = x.mean(dim=1)
        x = self.post_processing_module[f"{name}_output"](x)
        return x

    def build(self, batch):
        log.info(f"Built model {self.__class__.__name__}")
        image = batch["target_image"][0]
        challenge_images = batch["challenge_images"][0]
        image = torch.cat([image.unsqueeze(0), challenge_images], dim=0)
        text = batch["target_text"][0]

        image = self.preprocess_image(image)
        text = self.preprocess_text(text)

        if len(text.shape) == 1:
            text = text.unsqueeze(0)

        clip_output = self.model.forward(
            input_ids=text,
            pixel_values=image,
            output_hidden_states=True,
            return_loss=False,
        )

        image_hidden_token = clip_output.vision_model_output.hidden_states[-1]
        text_hidden_token = clip_output.text_model_output.hidden_states[-1]

        image_output = self.build_post_processing_module(
            name="image", x=image_hidden_token
        )
        text_output = self.build_post_processing_module(
            name="text", x=text_hidden_token
        )
        self.is_built = True

        return self.step(batch, batch_idx=0)

    def preprocess_image(self, image: torch.Tensor):
        image = image.cpu()
        if len(image.shape) == 4:
            image = image.unbind(0)
        image = self.processor(images=image, return_tensors="pt")["pixel_values"]
        image = image.to(self.model.device)

        if len(image.shape) != 4:
            raise ValueError(
                f"Input shape for class {self.__class__.__name__} in "
                "method forward_image must be 4, instead it is "
                f"{len(image.shape)}, for shape {image.shape}"
            )
        return image

    def preprocess_text(self, text: torch.Tensor) -> torch.Tensor:
        text = self.processor(
            text=text, return_tensors="pt", padding=True, truncation=True
        )["input_ids"]
        text = text.to(self.model.device)
        text = text.to(torch.int32)
        return text

    def forward_image(self, image: torch.Tensor) -> torch.Tensor:
        image = self.preprocess_image(image)
        clip_output = self.model.forward(pixel_values=image)

        image_hidden_token = clip_output.vision_model_output.hidden_states[-1]

        image_output = self.apply_post_processing(name="image", x=image_hidden_token)
        return image_output

    def forward_text(self, text: torch.Tensor) -> torch.Tensor:

        text = self.preprocess_text(text)
        if len(text.shape) == 1:
            text = text.unsqueeze(0)
        clip_output = self.model.forward(input_ids=text)

        text_hidden_token = clip_output.vision_model_output.hidden_states[-1]

        text_output = self.apply_post_processing(name="text", x=text_hidden_token)
        return text_output

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> CLIPOutput:

        image = self.preprocess_image(image)
        text = self.preprocess_text(text)

        if len(text.shape) == 1:
            text = text.unsqueeze(0)

        clip_output = self.model.forward(
            input_ids=text,
            pixel_values=image,
            output_hidden_states=True,
            return_loss=False,
        )

        image_hidden_token = clip_output.vision_model_output.hidden_states[-1]
        text_hidden_token = clip_output.text_model_output.hidden_states[-1]

        image_output = self.apply_post_processing(name="image", x=image_hidden_token)
        text_output = self.apply_post_processing(name="text", x=text_hidden_token)

        similarity = (
            torch.matmul(text_output, image_output.t()) * self.model.logit_scale
        )

        loss = contrastive_loss(similarity)

        return CLIPModelOutput(
            logits_per_image=similarity,
            image_embeds=image_output,
            text_embeds=text_output,
            loss=loss,
        )

    def step(self, batch, batch_idx):

        image = batch["target_image"][0]
        challenge_images = batch["challenge_images"][0]
        images = torch.cat([image.unsqueeze(0), challenge_images], dim=0)
        text = batch["target_text"][0]

        clip_output = self.forward(image=images, text=text)

        accuracy = (clip_output.logits_per_image.argmax(dim=-1) == 0).float().mean()
        output_dict = clip_output.__dict__
        output_dict["metrics"] = {"accuracy": accuracy, "loss": clip_output.loss}

        return output_dict["loss"], output_dict
