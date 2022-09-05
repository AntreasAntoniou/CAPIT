from __future__ import print_function

from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from clip.model import ModifiedResNet
from einops import rearrange, repeat
from omegaconf import ListConfig
from torch.nn import init

from tali.base import utils
from tali.config_repository import (
    AutoAveragerConfig,
    AutoCLIPResNetConfig,
    AutoCLIPTextTransformerConfig,
    AutoCLIPVisionTransformerConfig,
    AutoConv1DTransformersConfig,
    AutoVideoTransformersConfig,
)
from tali.models.auto_builder.base import (
    AvgPoolFlexibleDimension,
    BaseLinearOutputModel,
)
from tali.models.auto_builder.conv_modules import (
    ConvNetEmbedding,
    ConvPool1DStemBlock,
    ResNetBlock1D,
)
from transformers import (
    CLIPTextConfig,
    CLIPTextModel,
    CLIPVisionConfig,
)
from transformers.file_utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import (
    CLIP_VISION_INPUTS_DOCSTRING,
    CLIPEncoder,
    CLIPPreTrainedModel,
)

log = utils.get_logger(__name__)


class FCCNetwork(nn.Module):
    def __init__(
        self,
        num_hidden_features,
        embedding_output_features,
        num_hidden_layers,
        activation_fn=F.leaky_relu,
    ):
        super(FCCNetwork, self).__init__()
        self.layer_dict = nn.ModuleDict()
        self.is_layer_built = False
        self.num_hidden_features = num_hidden_features
        self.embedding_output_features = embedding_output_features
        self.num_hidden_layers = num_hidden_layers
        self.activation_fn = activation_fn

    def build(self, input_shape):
        out = torch.zeros(input_shape)

        for i in range(self.num_hidden_layers):
            self.layer_dict[f"fcc_layer_{i}"] = nn.Linear(
                in_features=out.shape[1],
                out_features=self.num_hidden_features,
                bias=True,
            )
            out = self.activation_fn(self.layer_dict[f"fcc_layer_{i}"].forward(out))

        self.layer_dict["fcc_layer_output"] = nn.Linear(
            in_features=out.shape[1],
            out_features=self.embedding_output_features,
            bias=True,
        )

        out = self.layer_dict["fcc_layer_output"].forward(out)

        self.is_layer_built = True

        log.debug(
            f"Build {self.__class__.__name__} with input shape {input_shape} with "
            f"output shape {out.shape}"
        )

    def forward(self, x):
        if not self.is_layer_built:
            self.build(input_shape=x.shape)
            self.type_as(x)

        out = x

        self.type_as(out)

        for i in range(self.num_hidden_layers):
            out = self.activation_fn(self.layer_dict[f"fcc_layer_{i}"].forward(out))

        out = self.layer_dict["fcc_layer_output"].forward(out)

        return out


class ChooseSpecificTimeStepFromVector(nn.Module):
    def __init__(self, time_step_to_choose):
        super(ChooseSpecificTimeStepFromVector, self).__init__()
        self.time_step_to_choose = time_step_to_choose

    def forward(self, x):

        return x, x[:, self.time_step_to_choose, :]


class Conv1DTransformer(nn.Module):
    def __init__(
        self,
        grid_patch_size: int,
        resnet_num_filters: int,
        resnet_num_stages: int,
        resnet_num_blocks: int,
        resnet_kernel_size: int,
        resnet_dilated: bool,
        transformer_num_filters: int,
        transformer_num_layers: int,
        transformer_num_heads: int,
        transformer_dim_feedforward: int,
        return_sequence_idx: int = None,
        stem_conv_bias: bool = False,
    ):
        super(Conv1DTransformer, self).__init__()
        self.grid_patch_size = grid_patch_size
        self.resnet_num_filters = resnet_num_filters
        self.resnet_num_stages = resnet_num_stages
        self.resnet_num_blocks = resnet_num_blocks
        self.resnet_kernel_size = resnet_kernel_size
        self.resnet_dilated = resnet_dilated
        self.transformer_num_filters = transformer_num_filters
        self.transformer_num_layers = transformer_num_layers
        self.transformer_num_heads = transformer_num_heads
        self.transformer_dim_feedforward = transformer_dim_feedforward
        self.stem_conv_bias = stem_conv_bias
        self.return_sequence_idx = return_sequence_idx

        self.is_built = False

    def build(self, input_shape):
        dummy_x = torch.zeros(input_shape)

        out = dummy_x

        self.layer_dict = nn.ModuleDict()

        self.layer_dict["conv1d_resnet"] = ConvNetEmbedding(
            num_filters=self.resnet_num_filters,
            num_stages=self.resnet_num_stages,
            num_blocks=self.resnet_num_blocks,
            dilated=self.resnet_dilated,
            kernel_size=self.resnet_kernel_size,
            stem_block_type=ConvPool1DStemBlock,
            processing_block_type=ResNetBlock1D,
            reduction_block_type=ResNetBlock1D,
        )

        out = self.layer_dict["conv1d_resnet"].forward(out)

        self.layer_dict["conv1d_activation"] = nn.LeakyReLU(0.1)

        out = self.layer_dict["conv1d_activation"].forward(out)
        log.debug(f"conv1d_resnet output shape {out.shape}")

        self.layer_dict["conv_patch_cutter"] = nn.Conv1d(
            in_channels=out.shape[1],
            out_channels=self.transformer_num_filters,
            kernel_size=self.grid_patch_size,
            stride=self.grid_patch_size,
        )

        out = self.layer_dict["conv_patch_cutter"].forward(out)
        log.debug(f"conv_feature_pooling output shape {out.shape}")

        self.layer_dict["conv_patch_cutter_activation"] = nn.LeakyReLU(0.1)

        out = self.layer_dict["conv_patch_cutter_activation"].forward(out)

        out = rearrange(out, "b f h -> (b h) f")

        # self.layer_dict["stem_layer_normalization"] = nn.LayerNorm(out.shape[1])

        # out = self.layer_dict["stem_layer_normalization"].forward(out)
        # log.debug(f"stem_layer_normalization output shape {out.shape}")

        out = rearrange(out, "(b s) (f) -> b s f", b=dummy_x.shape[0])
        log.debug(f"rearrange output shape {out.shape}")

        self.positional_embeddings = nn.Parameter(
            data=torch.empty((out.shape[1], self.transformer_num_filters)),
            requires_grad=True,
        )

        init.normal_(self.positional_embeddings)

        positional_embeddings = repeat(
            self.positional_embeddings, "p f -> b p f", b=dummy_x.shape[0]
        )

        out = out + positional_embeddings
        log.debug(f"out + positional_embeddings output shape {out.shape}")

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformer_num_filters,
            dim_feedforward=self.transformer_dim_feedforward,
            nhead=self.transformer_num_heads,
            activation=F.gelu,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.layer_dict["transformer_encoder"] = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.transformer_num_layers,
        )

        out = self.layer_dict["transformer_encoder"].forward(out)
        if self.return_sequence_idx is not None:
            out = out[:, self.return_sequence_idx, :]
        log.debug(f"transformer_encoder output shape {out.shape}")

        self.is_built = True
        log.debug(
            f"Build {self.__class__.__name__} with input shape {input_shape} with "
            f"output shape {out.shape}"
        )

    def forward(self, x):
        if not self.is_built:
            self.build(input_shape=x.shape)

        # log.info(f"{x.mean()}, {x.std()}, {x.max()}, {x.min()}")

        out = x

        out = self.layer_dict["conv1d_resnet"].forward(out)

        out = self.layer_dict["conv1d_activation"].forward(out)

        out = self.layer_dict["conv_patch_cutter"].forward(out)

        out = self.layer_dict["conv_patch_cutter_activation"].forward(out)

        out = rearrange(out, "b f h -> b h f")

        positional_embeddings = repeat(
            self.positional_embeddings, "p f -> b p f", b=x.shape[0]
        )

        out = out + positional_embeddings

        out = self.layer_dict["transformer_encoder"].forward(out)

        if self.return_sequence_idx is not None:
            out = out[:, self.return_sequence_idx, :]

        return out


class VideoTransformer(nn.Module):
    def __init__(
        self,
        transformer_num_filters: int,
        transformer_num_layers: int,
        transformer_num_heads: int,
        transformer_dim_feedforward: int,
        image_embedding: nn.Identity(),
    ):
        super(VideoTransformer, self).__init__()
        self.transformer_num_filters = transformer_num_filters
        self.transformer_num_layers = transformer_num_layers
        self.transformer_num_heads = transformer_num_heads
        self.transformer_dim_feedforward = transformer_dim_feedforward

        self.image_embedding = image_embedding
        self.layer_dict = nn.ModuleDict()
        self.layer_params = nn.ParameterDict()
        self.is_built = False

    def build(self, input_shape):
        dummy_x = torch.zeros(input_shape)

        out = dummy_x.view(-1, dummy_x.shape[-3], dummy_x.shape[-2], dummy_x.shape[-1])

        _, out = self.image_embedding(out)

        out = out.view(dummy_x.shape[0] * dummy_x.shape[1], -1)

        self.linear_projection = nn.Linear(
            out.shape[-1], self.transformer_num_filters, bias=True
        )

        out = self.linear_projection.forward(out)

        out = out.view(dummy_x.shape[0], dummy_x.shape[1], -1)

        self.positional_embeddings = nn.Parameter(
            torch.empty(size=(out.shape[1], out.shape[2]))
        )

        nn.init.normal(self.positional_embeddings)

        out = out + repeat(self.positional_embeddings, "p f -> b p f", b=out.shape[0])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformer_num_filters,
            dim_feedforward=self.transformer_dim_feedforward,
            nhead=self.transformer_num_heads,
            batch_first=True,
        )

        self.layer_dict["transformer_encoder"] = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.transformer_num_layers,
        )

        out = self.layer_dict["transformer_encoder"].forward(out)

        self.is_built = True

        log.debug(
            f"Build {self.__class__.__name__} with input shape {input_shape} with "
            f"output shape {out.shape}"
        )

    def forward(self, x):

        if not self.is_built:
            self.build(input_shape=x.shape)

        out = x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1])

        _, out = self.image_embedding(out)

        out = out.view(x.shape[0] * x.shape[1], -1)

        out = self.linear_projection.forward(out)

        out = out.view(x.shape[0], x.shape[1], -1)

        out = out + repeat(self.positional_embeddings, "p f -> b p f", b=out.shape[0])

        out = self.layer_dict["transformer_encoder"].forward(out)

        return out

    def connect_image_embedding(self, image_embedding):
        self.image_embedding = image_embedding


class Averager(nn.Module):
    def __init__(self, image_embedding: nn.Identity()):
        super(Averager, self).__init__()

        self.image_embedding = image_embedding
        self.is_built = False

    def build(self, input_shape):
        dummy_x = torch.zeros(input_shape)

        out = dummy_x.view(-1, dummy_x.shape[-3], dummy_x.shape[-2], dummy_x.shape[-1])

        if self.image_embedding is not None:
            _, out = self.image_embedding(out)

        out = out.view(dummy_x.shape[0], dummy_x.shape[1], -1)

        self.is_built = True

        log.debug(
            f"Build {self.__class__.__name__} with input shape {input_shape} with "
            f"output shape {out.shape}"
        )

    def forward(self, x):

        if not self.is_built:
            self.build(input_shape=x.shape)

        out = x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1])

        if self.image_embedding is not None:
            _, out = self.image_embedding(out)

        out = out.view(x.shape[0], x.shape[1], -1)

        return out

    def connect_image_embedding(self, image_embedding):
        self.image_embedding = image_embedding


class AutoConv1DTransformers(BaseLinearOutputModel):
    def __init__(self, config: AutoConv1DTransformersConfig):
        feature_embedding_modules = [
            nn.InstanceNorm1d,
            Conv1DTransformer,
        ]
        feature_embeddings_args = [
            dict(num_features=2, affine=True, track_running_stats=True),
            dict(
                grid_patch_size=config.grid_patch_size,
                resnet_num_filters=config.resnet_num_filters,
                resnet_num_stages=config.resnet_num_stages,
                resnet_num_blocks=config.resnet_num_blocks,
                resnet_kernel_size=config.resnet_kernel_size,
                resnet_dilated=config.resnet_dilated,
                transformer_num_filters=config.transformer_num_filters,
                transformer_num_layers=config.transformer_num_layers,
                transformer_num_heads=config.transformer_num_heads,
                transformer_dim_feedforward=config.transformer_dim_feedforward,
                stem_conv_bias=True,
                return_sequence_idx=0,
            ),
        ]
        super(AutoConv1DTransformers, self).__init__(
            embedding_output_features=config.embedding_output_features,
            feature_embedding_module_list=feature_embedding_modules,
            feature_embedding_args=feature_embeddings_args,
        )


class AutoVideoTransformers(BaseLinearOutputModel):
    def __init__(
        self,
        config: AutoVideoTransformersConfig,
        image_embedding: Optional[nn.Module] = None,
    ):
        self.config = config
        feature_embedding_modules = [
            VideoTransformer,
            AvgPoolFlexibleDimension,
        ]

        feature_embeddings_args = [
            dict(
                image_embedding=image_embedding,
                transformer_num_filters=config.transformer_num_filters,
                transformer_num_layers=config.transformer_num_layers,
                transformer_num_heads=config.transformer_num_heads,
                transformer_dim_feedforward=config.transformer_dim_feedforward,
            ),
            dict(dim=1),
        ]
        super(AutoVideoTransformers, self).__init__(
            embedding_output_features=config.embedding_output_features,
            feature_embedding_module_list=feature_embedding_modules,
            feature_embedding_args=feature_embeddings_args,
        )

    def connect_image_embedding(self, image_embedding: nn.Module):
        self.__init__(config=self.config, image_embedding=image_embedding)


class AutoAverager(BaseLinearOutputModel):
    def __init__(
        self,
        config: AutoAveragerConfig,
        image_embedding: Optional[nn.Module] = None,
    ):
        self.config = config
        feature_embedding_modules = [
            Averager,
            AvgPoolFlexibleDimension,
        ]

        feature_embeddings_args = [
            dict(image_embedding=image_embedding),
            dict(dim=config.dim),
        ]
        super(AutoAverager, self).__init__(
            embedding_output_features=config.embedding_output_features,
            feature_embedding_module_list=feature_embedding_modules,
            feature_embedding_args=feature_embeddings_args,
        )

    def connect_image_embedding(self, image_embedding: nn.Module):
        self.__init__(config=self.config, image_embedding=image_embedding)


class AutoCLIPVisionTransformer(BaseLinearOutputModel):
    def __init__(self, config: AutoCLIPVisionTransformerConfig):
        feature_embedding_modules = [
            CLIPVisionModel,
        ]
        log.info(f"Building {self.__class__.__name__} with config: {config}")
        feature_embeddings_args = [
            dict(
                config=CLIPVisionConfig(
                    hidden_size=config.transformer_num_filters,
                    intermediate_size=config.transformer_dim_feedforward,
                    num_hidden_layers=config.transformer_num_layers,
                    num_attention_heads=config.transformer_num_heads,
                    image_size=config.image_resolution,
                    patch_size=config.grid_patch_size,
                    hidden_act="quick_gelu",
                    layer_norm_eps=0.00001,
                    dropout=0.0,
                    attention_dropout=0.0,
                    initializer_range=0.02,
                    initializer_factor=1.0,
                )
            ),
        ]
        super(AutoCLIPVisionTransformer, self).__init__(
            embedding_output_features=config.embedding_output_features,
            feature_embedding_module_list=feature_embedding_modules,
            feature_embedding_args=feature_embeddings_args,
        )


class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        from rich import print

        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        if isinstance(self.image_size, (Tuple, List, ListConfig)):
            self.num_patches = int(
                torch.prod(
                    torch.Tensor(
                        [image_dim // self.patch_size for image_dim in self.image_size]
                    )
                ).item()
            )
        else:
            self.num_patches = (self.image_size // self.patch_size) ** 2

        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids", torch.arange(self.num_positions).expand((1, -1))
        )

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(
            pixel_values
        )  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class CLIPVisionTransformer(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim)
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim)

    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=BaseModelOutputWithPooling, config_class=CLIPVisionConfig
    )
    def forward(
        self,
        pixel_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:

        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class CLIPVisionModel(CLIPPreTrainedModel):
    config_class = CLIPVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)
        self.vision_model = CLIPVisionTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=BaseModelOutputWithPooling, config_class=CLIPVisionConfig
    )
    def forward(
        self,
        pixel_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import CLIPProcessor, CLIPVisionModel

        >>> model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class AttentionPool2dNonSquare(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
        embed_dim: int,
        num_heads: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(height * width + 1, embed_dim) / embed_dim**0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(
            2, 0, 1
        )  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )

        return x[0]


class ModifiedResNetNonSquareImages(ModifiedResNet):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super(ModifiedResNetNonSquareImages, self).__init__(
            layers, output_dim, heads, input_resolution[0], width
        )

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2dNonSquare(
            height=input_resolution[0] // 32,
            width=input_resolution[1] // 32,
            embed_dim=embed_dim,
            num_heads=heads,
            output_dim=output_dim,
        )


class AutoCLIPResNet(BaseLinearOutputModel):
    def __init__(self, config: AutoCLIPResNetConfig):
        log.info(f"Building {self.__class__.__name__} with config: {config}")

        vision_heads = config.vision_width * 32 // 64

        feature_embedding_modules = [
            # nn.InstanceNorm2d,
            ModifiedResNetNonSquareImages,
        ]

        feature_embeddings_args = [
            # dict(num_features=3, affine=True, track_running_stats=True),
            dict(
                layers=config.vision_layers,
                output_dim=config.embedding_output_features,
                heads=vision_heads,
                input_resolution=config.image_resolution,
                width=config.vision_width,
            ),
        ]
        super(AutoCLIPResNet, self).__init__(
            embedding_output_features=config.embedding_output_features,
            feature_embedding_module_list=feature_embedding_modules,
            feature_embedding_args=feature_embeddings_args,
        )


class MeanLayer(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x):
        x = x.view(x.shape[0], -1, x.shape[-1])
        x = x.mean(dim=1)
        x = x.view(x.shape[0], -1)
        return x


class AutoDumbNet(BaseLinearOutputModel):
    def __init__(self, embedding_output_features):
        feature_embedding_modules = [MeanLayer]

        feature_embeddings_args = [
            dict(),
        ]
        super(AutoDumbNet, self).__init__(
            embedding_output_features=embedding_output_features,
            feature_embedding_module_list=feature_embedding_modules,
            feature_embedding_args=feature_embeddings_args,
        )


class AutoCLIPTextTransformer(BaseLinearOutputModel):
    def __init__(self, config: AutoCLIPTextTransformerConfig):
        text_config = CLIPTextConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.transformer_num_filters,
            intermediate_size=config.transformer_dim_feedforward,
            num_hidden_layers=config.transformer_num_layers,
            num_attention_heads=config.transformer_num_heads,
            max_position_embeddings=config.context_length,
            hidden_act="quick_gelu",
            layer_norm_eps=0.00001,
            dropout=0.0,
            attention_dropout=0.0,
            initializer_range=0.02,
            initializer_factor=1.0,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
        )
        feature_embedding_modules = [CLIPTextModel]

        feature_embeddings_args = [dict(config=text_config)]

        super(AutoCLIPTextTransformer, self).__init__(
            embedding_output_features=config.embedding_output_features,
            feature_embedding_module_list=feature_embedding_modules,
            feature_embedding_args=feature_embeddings_args,
            input_type="long",
        )
