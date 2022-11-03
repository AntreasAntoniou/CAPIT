import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from capit.base.utils import get_logger
from capit.data.tokenizers import HuggingFaceBPETokenizer
from dotted_dict import DottedDict
from rich import print
from torchvision.transforms.functional import normalize
from transformers import CLIPModel, CLIPProcessor
from transformers.models.clip.modeling_clip import contrastive_loss

from capit.decorators import configurable

log = get_logger(__name__)


def resize_custom(image, target_image_shape, interpolation="bilinear", debug=False):
    """
    Resize an image to a target size.
    Parameters
    ----------
    image
    target_image_shape
    interpolation

    Returns
    -------

    """
    target_w = target_image_shape[1]
    target_h = target_image_shape[2]

    current_w = image.shape[2]
    current_h = image.shape[3]

    if current_w > target_w:
        image = image[:, :, :target_w]
        if debug:
            print(
                f"Condition met: current_w > target_w: Resized image from {current_w} to {target_w} == {image.shape}"
            )

    if current_h > target_h:
        image = image[:, :, :, :target_h]
        if debug:
            print(
                f"Condition met: current_h > target_h: Resized image from {current_h} to {target_h} == {image.shape}"
            )

    if current_w < target_w:
        pad_size = int(np.floor((target_w - current_w) / 2))
        p2dw = (0, 0, pad_size, pad_size)
        image = F.pad(image, p2dw, "constant", 0)
        if debug:
            print(
                f"Condition met: current_w < target_w: Resized image from {current_w} to {target_w} == {image.shape}"
            )

    if current_h < target_h:
        pad_size = int(np.floor((target_h - current_h) / 2))
        p2dh = (pad_size, pad_size, 0, 0)
        image = F.pad(image, p2dh, "constant", 0)
        if debug:
            print(
                f"Condition met: current_h < target_h: Resized image from {current_h} to {target_h} == {image.shape}"
            )

    return image


@configurable
class CLIPImageTextModel(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        pretrained: bool = True,
        fine_tunable: bool = True,
    ):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name_or_path)
        self.processor = CLIPProcessor.from_pretrained(model_name_or_path)
        self.tokenizer = HuggingFaceBPETokenizer(context_length=77, parallel=True)

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
        self.mean = self.processor.feature_extractor.image_mean
        self.std = self.processor.feature_extractor.image_std
        # self.input_normalization = nn.InstanceNorm2d(
        #     num_features=3,
        #     affine=True,
        # )

    def build(self, batch):
        log.info(f"Built model {self.__class__.__name__}")
        return self.step(batch, 0)

    def preprocess_image(self, image: torch.Tensor):
        # if self.fine_tunable is True:
        #     image = self.input_normalization(image)

        image = normalize(image, mean=self.mean, std=self.std)

        if len(image.shape) != 4:
            raise ValueError(
                f"Input shape for class {self.__class__.__name__} in "
                f"method forward_image must be 4, instead it is "
                f"{len(image.shape)}, for shape {image.shape}"
            )
        return image

    def proprocess_text(self, text: torch.Tensor):
        text = self.tokenizer.forward(x=text)
        text = text.to(self.model.device)
        text = text.to(torch.int32)
        return text

    def forward_image(self, image: torch.Tensor):
        image = image.to(self.model.device)
        image = self.preprocess_image(image)
        return self.model.get_image_features(image)

    def forward_text(self, text: torch.Tensor):

        text = self.proprocess_text(text)
        if len(text.shape) == 1:
            text = text.unsqueeze(0)
        return self.model.get_text_features(text)

    def forward(self, image: torch.Tensor, text: torch.Tensor):

        image_embeds = self.forward_image(image)
        text_embeds = self.forward_text(text)

        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits

        logit_scale = self.model.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.T

        return DottedDict(
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
        )

    def predict_individual(self, image: torch.Tensor, text: torch.Tensor):
        image_embeds = self.forward_image(image)
        text_embeds = self.forward_text(text)

        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits

        logit_scale = self.model.logit_scale.exp()
        logits_per_text = (
            torch.sum(text_embeds * image_embeds.t(), dim=1)
        ) * logit_scale
        logits_per_image = logits_per_text.T

        return DottedDict(
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
        )

    def step(self, batch, batch_idx):
        image = batch["image"][0]
        query_images = batch["query_image_set"][0]
        images = torch.cat([image.unsqueeze(0), query_images], dim=0)
        text = batch["text"][0]
        output_dict = self.forward(images, text)
        opt_loss = contrastive_loss(output_dict.logits_per_text)
        accuracy = (output_dict.logits_per_text.argmax(dim=-1) == 0).float().mean()
        output_dict.metrics = DottedDict()
        output_dict.metrics.accuracy = accuracy
        output_dict.metrics.loss = opt_loss

        return opt_loss, output_dict
