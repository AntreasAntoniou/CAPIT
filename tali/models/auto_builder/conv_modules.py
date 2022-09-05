from __future__ import print_function
from __future__ import print_function

import logging
from collections import OrderedDict
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from tali.base.utils import get_logger

log = get_logger(__name__)


class ClassificationModel(nn.Module):
    def __init__(
        self,
        feature_embedding_module_list: Union[List[nn.Module], nn.Module],
        feature_embedding_args: Union[List[dict], dict],
        num_classes: int,
        input_type: torch.float32,
    ):
        self.feature_embedding_module_list = feature_embedding_module_list
        self.feature_embedding_args = feature_embedding_args
        self.num_classes = num_classes
        self.is_layer_built = False
        self.input_type = input_type
        super(ClassificationModel, self).__init__()

    def build(self, input_shape):
        """
        Builds network whilst automatically inferring shapes of layers.
        """
        out = torch.zeros(input_shape, dtype=self.input_type)
        logging.debug(
            "Building basic block of a classification model using input shape"
        )
        # assumes that input shape is b, c, h, w
        logging.debug(f"build input {out.shape}")

        if isinstance(self.feature_embedding_module_list, list):
            self.feature_embedding_module = nn.Sequential(
                OrderedDict(
                    [
                        (str(idx), item(**item_args))
                        for idx, (item, item_args) in enumerate(
                            zip(
                                self.feature_embedding_module_list,
                                self.feature_embedding_args,
                            )
                        )
                    ]
                )
            )
        else:
            self.feature_embedding_module = self.feature_embedding_module_list(
                **self.feature_embedding_args
            )

        out = self.feature_embedding_module.forward(out)
        if isinstance(out, tuple):
            out = out[-1]

        out = out.view(out.shape[0], -1)
        logging.debug(f"build input {out.shape}")

        # classification head
        logging.debug(f"{out.shape[1]} {self.num_classes}")
        self.output_layer = nn.Linear(
            in_features=out.shape[1], out_features=self.num_classes, bias=True
        )
        out = self.output_layer.forward(out)
        logging.debug(f"build input {out.shape}")

        self.is_layer_built = True
        logging.debug("Summary:")
        logging.debug(
            f"Build {self.__class__.__name__} with input shape {input_shape} with "
            f"output shape {out.shape}"
        )
        # for layer_idx, layer_params in self.named_parameters():
        #     print(layer_idx, layer_params.shape)

    def forward(self, x):
        """
        Forward propagates the network given an input batch
        :param x: Inputs x (b, c, w, h)
        :return: preds (b, num_classes)
        """
        x = x.type(self.input_type)
        if not self.is_layer_built:
            self.build(input_shape=x.shape)
            self.to(x.device)

        out = x
        features = self.feature_embedding_module.forward(out)

        if isinstance(features, tuple):
            features = features[-1]

        out = features.view(features.shape[0], -1)
        out = self.output_layer.forward(out)
        return out, features


class ResNet(nn.Module):
    """https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']
    https://github.com/pytorch/vision/blob/331f126855a106d29e1de23f8bbc3cde66c603e5/
    torchvision/architectures/resnet.py#L144
    https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    model_ft.fc = nn.Linear(num_ftrs, num_classes)

    """

    def __init__(self, model_name_to_download: str, pretrained: bool = True):
        super(ResNet, self).__init__()
        self.pretrained = pretrained
        self.model_name_to_download = model_name_to_download
        self.is_layer_built = False

    def build(self, input_shape):
        x_dummy = torch.zeros(input_shape)
        out = x_dummy

        self.linear_transform = nn.Conv2d(
            in_channels=out.shape[1],
            out_channels=3,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            dilation=(1, 1),
            groups=1,
            bias=False,
            padding_mode="zeros",
        )

        self.model = torch.hub.load(
            "pytorch/vision:v0.6.0",
            self.model_name_to_download,
            pretrained=self.pretrained,
        )

        self.model.fc = nn.Identity()

        self.model.avgpool = nn.Identity()

        out = self.linear_transform.forward(out)

        out = self.model.forward(out)

        out = out.view(out.shape[0], 512, -1)

        out = out.view(
            out.shape[0],
            512,
            int(np.sqrt(out.shape[-1])),
            int(np.sqrt(out.shape[-1])),
        )

        self.is_layer_built = True

        logging.debug(
            f"Build {self.__class__.__name__} with input shape {input_shape} with "
            f"output shape {out.shape}"
        )

    def forward(self, x):
        if not self.is_layer_built:
            self.build(input_shape=x.shape)
            self.to(x.device)

        out = x

        out = self.linear_transform.forward(out)
        # out shape (b*s, channels)
        out = self.model.forward(out)

        out = out.view(out.shape[0], 512, -1)

        out = out.view(
            out.shape[0],
            512,
            int(np.sqrt(out.shape[-1])),
            int(np.sqrt(out.shape[-1])),
        )

        return out


class BaseStyleLayer(nn.Module):
    def __init__(self):
        super(BaseStyleLayer, self).__init__()
        self.is_layer_built = False

    def build(self, input_shape):
        out = torch.zeros(input_shape)

        self.is_layer_built = True

        logging.debug(
            f"Build {self.__class__.__name__} with input shape {input_shape} with "
            f"output shape {out.shape}"
        )

    def forward(self, x):
        if not self.is_layer_built:
            self.build(input_shape=x.shape)
            self.to(x.device)

        return x


class FullyConnectedLayer(nn.Module):
    def __init__(self, num_filters, bias):
        super(FullyConnectedLayer, self).__init__()
        self.num_filters = num_filters
        self.bias = bias
        self.is_layer_built = False

    def build(self, input_shape):
        out = torch.zeros(input_shape)

        self.fc1 = nn.Linear(
            in_features=int(np.prod(out.shape[1:])),
            out_features=self.num_filters,
            bias=self.bias,
        )
        out = self.fc1.forward(out)

        self.is_layer_built = True

        logging.debug(
            f"Build {self.__class__.__name__} with input shape {input_shape} with "
            f"output shape {out.shape}"
        )

    def forward(self, x):
        assert (
            len(x.shape) == 2
        ), "input tensor should have a length of 2 but its instead {}".format(x.shape)

        if not self.is_layer_built:
            self.build(input_shape=x.shape)
            self.to(x.device)

        x = self.fc1(x)

        return x


class Conv2dBNLeakyReLU(nn.Module):
    def __init__(
        self,
        out_channels: int,
        kernel_size: Tuple[int],
        stride: Tuple[int],
        padding: int,
        dilation: Tuple[int] = (1, 1),
        bias: bool = False,
    ):
        super(Conv2dBNLeakyReLU, self).__init__()
        self.is_layer_built = False
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

    def build(self, input_shape):
        out = torch.zeros(input_shape)

        self.conv = nn.Conv2d(
            in_channels=input_shape[1],
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=1,
            bias=self.bias,
            padding_mode="zeros",
        )

        out = self.conv.forward(out)

        self.bn = nn.BatchNorm2d(
            track_running_stats=True,
            affine=True,
            num_features=out.shape[1],
            eps=1e-5,
        )

        out = self.bn.forward(out)

        out = F.leaky_relu(out)

        self.is_layer_built = True

        logging.debug(
            f"Build {self.__class__.__name__} with input shape {input_shape} with "
            f"output shape {out.shape}"
        )

    def forward(self, x):
        if not self.is_layer_built:
            self.build(input_shape=x.shape)
            self.to(x.device)

        out = x

        out = self.conv.forward(out)

        out = self.bn.forward(out)

        out = F.leaky_relu(out)

        return out


class Conv1dBNLeakyReLU(nn.Module):
    def __init__(
        self,
        out_channels: int,
        kernel_size: Tuple[int],
        stride: Tuple[int],
        padding: int,
        dilation: Tuple[int] = (1,),
        bias: bool = False,
    ):
        super(Conv1dBNLeakyReLU, self).__init__()
        self.is_layer_built = False
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

    def build(self, input_shape):
        out = torch.zeros(input_shape)

        self.conv = nn.Conv1d(
            in_channels=input_shape[1],
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=1,
            bias=self.bias,
            padding_mode="zeros",
        )

        out = self.conv.forward(out)

        self.bn = nn.BatchNorm1d(
            track_running_stats=True,
            affine=True,
            num_features=out.shape[1],
            eps=1e-5,
        )

        out = self.bn.forward(out)

        out = F.leaky_relu(out)

        self.is_layer_built = True

        logging.debug(
            f"Build {self.__class__.__name__} with input shape {input_shape} with "
            f"output shape {out.shape}"
        )

    def forward(self, x):
        if not self.is_layer_built:
            self.build(input_shape=x.shape)
            self.to(x.device)

        out = x

        out = self.conv.forward(out)

        out = self.bn.forward(out)

        out = F.leaky_relu(out)

        return out


class SqueezeExciteConv2dBNLeakyReLU(nn.Module):
    def __init__(
        self,
        out_channels: int,
        kernel_size: Tuple[int],
        stride: Tuple[int],
        padding: int,
        dilation: Tuple[int] = (1, 1),
        bias: bool = False,
    ):
        super(SqueezeExciteConv2dBNLeakyReLU, self).__init__()
        self.is_layer_built = False
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

    def build(self, input_shape):
        out = torch.zeros(input_shape)

        pooled = (
            F.avg_pool2d(input=out, kernel_size=out.shape[-1]).squeeze(3).squeeze(2)
        )

        self.attention_fcc_network_in = nn.Linear(
            in_features=pooled.shape[-1], out_features=64, bias=True
        )

        attention_out = self.attention_fcc_network_in.forward(input=pooled)

        attention_out = F.leaky_relu(attention_out)

        self.attention_fcc_network_out = nn.Linear(
            in_features=attention_out.shape[-1],
            out_features=pooled.shape[-1],
            bias=True,
        )

        attention_out = self.attention_fcc_network_out.forward(attention_out)

        attention_out = F.sigmoid(attention_out)

        out = out * attention_out.unsqueeze(2).unsqueeze(3)

        self.conv = nn.Conv2d(
            in_channels=input_shape[1],
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=1,
            bias=self.bias,
            padding_mode="zeros",
        )

        out = self.conv.forward(out)

        self.bn = nn.BatchNorm2d(
            track_running_stats=True,
            affine=True,
            num_features=out.shape[1],
            eps=1e-5,
        )

        out = self.bn.forward(out)

        out = F.leaky_relu(out)

        self.is_layer_built = True

        logging.debug(
            f"Build {self.__class__.__name__} with input shape {input_shape} with "
            f"output shape {out.shape}"
        )

    def forward(self, x):
        if not self.is_layer_built:
            self.build(input_shape=x.shape)
            self.to(x.device)

        out = x

        pooled = (
            F.avg_pool2d(input=out, kernel_size=out.shape[-1]).squeeze(3).squeeze(2)
        )

        attention_out = self.attention_fcc_network_in.forward(input=pooled)

        attention_out = F.leaky_relu(attention_out)

        attention_out = self.attention_fcc_network_out.forward(attention_out)

        attention_out = F.sigmoid(attention_out)

        out = out * attention_out.unsqueeze(2).unsqueeze(3)

        out = self.conv.forward(out)

        out = self.bn.forward(out)

        out = F.leaky_relu(out)

        return out


class SqueezeExciteConv1dBNLeakyReLU(nn.Module):
    def __init__(
        self,
        out_channels: int,
        kernel_size: Tuple[int],
        stride: Tuple[int],
        padding: int,
        dilation: Tuple[int] = (1,),
        bias: bool = False,
    ):
        super(SqueezeExciteConv1dBNLeakyReLU, self).__init__()
        self.is_layer_built = False
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

    def build(self, input_shape):
        out = torch.zeros(input_shape)

        pooled = F.avg_pool1d(input=out, kernel_size=out.shape[-1]).squeeze(2)

        self.attention_fcc_network_in = nn.Linear(
            in_features=pooled.shape[-1], out_features=64, bias=True
        )

        attention_out = self.attention_fcc_network_in.forward(input=pooled)

        attention_out = F.leaky_relu(attention_out)

        self.attention_fcc_network_out = nn.Linear(
            in_features=attention_out.shape[-1],
            out_features=pooled.shape[-1],
            bias=True,
        )

        attention_out = self.attention_fcc_network_out.forward(attention_out)

        attention_out = F.sigmoid(attention_out)

        out = out * attention_out.unsqueeze(2)

        self.conv = nn.Conv1d(
            in_channels=input_shape[1],
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=1,
            bias=self.bias,
            padding_mode="zeros",
        )

        out = self.conv.forward(out)

        self.bn = nn.BatchNorm1d(
            track_running_stats=True,
            affine=True,
            num_features=out.shape[1],
            eps=1e-5,
        )

        out = self.bn.forward(out)

        out = F.leaky_relu(out)

        self.is_layer_built = True

        logging.debug(
            f"Build {self.__class__.__name__} with input shape {input_shape} with "
            f"output shape {out.shape}"
        )

    def forward(self, x):
        if not self.is_layer_built:
            self.build(input_shape=x.shape)
            self.to(x.device)

        out = x

        pooled = F.avg_pool1d(input=out, kernel_size=out.shape[-1]).squeeze(2)

        attention_out = self.attention_fcc_network_in.forward(input=pooled)

        attention_out = F.leaky_relu(attention_out)

        attention_out = self.attention_fcc_network_out.forward(attention_out)

        attention_out = F.sigmoid(attention_out)

        out = out * attention_out.unsqueeze(2)

        out = self.conv.forward(out)

        out = self.bn.forward(out)

        out = F.leaky_relu(out)

        return out


class Conv2dEmbedding(nn.Module):
    def __init__(
        self,
        layer_filter_list: List[int],
        kernel_size: Tuple[int],
        stride: Tuple[int],
        padding: int,
        avg_pool_kernel_size: Tuple[int],
        avg_pool_stride: Tuple[int],
        dilation: Tuple[int] = (1,),
        bias: bool = False,
    ):
        super(Conv2dEmbedding, self).__init__()
        self.is_layer_built = False
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        self.layer_filter_list = layer_filter_list
        self.avg_pool_kernel_size = avg_pool_kernel_size
        self.avg_pool_stride = avg_pool_stride

    def build(self, input_shape):
        out = torch.zeros(input_shape)
        self.layer_dict = nn.ModuleDict()

        for idx, num_filters in enumerate(self.layer_filter_list):
            self.layer_dict["conv_{}".format(idx)] = Conv2dBNLeakyReLU(
                out_channels=num_filters,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                bias=self.bias,
            )
            out = self.layer_dict["conv_{}".format(idx)].forward(out)

            out = F.avg_pool2d(
                input=out,
                kernel_size=self.avg_pool_kernel_size,
                stride=self.avg_pool_stride,
            )  # TODO: avg_pool kernel_size, stride should be set as parameters

        self.is_layer_built = True

        logging.debug(
            f"Build {self.__class__.__name__} with input shape {input_shape} with "
            f"output shape {out.shape}"
        )

        for layer_idx, layer_params in self.named_parameters():
            logging.debug(f"{layer_idx} layer_params.shape")

    def forward(self, x):

        if not self.is_layer_built:
            self.build(input_shape=x.shape)
            self.to(x.device)

        out = x

        for idx, num_filters in enumerate(self.layer_filter_list):
            out = self.layer_dict["conv_{}".format(idx)].forward(out)
            out = F.avg_pool2d(
                input=out,
                kernel_size=self.avg_pool_kernel_size,
                stride=self.avg_pool_stride,
            )

        return out


class ConcatenateLayer(nn.Module):
    def __init__(self, dim: int):
        super(ConcatenateLayer, self).__init__()
        self.dim = dim

    def forward(self, x: list):
        x = torch.cat(tensors=x, dim=self.dim)
        return x


class AvgPoolSpatialAndSliceIntegrator(nn.Module):
    def __init__(self):
        super(AvgPoolSpatialAndSliceIntegrator, self).__init__()
        self.is_layer_built = False

    def build(self, input_shape):
        out = torch.zeros(input_shape)
        # b, s, out_features, w / n, h / n
        out = out.mean(dim=4)
        # b, s, out_features, w / n
        out = out.mean(dim=3)
        # b, s, out_features
        out = out.mean(dim=1)
        # b, out_features

        self.is_layer_built = True

        logging.debug(
            f"Build {self.__class__.__name__} with input shape {input_shape} with "
            f"output shape {out.shape}"
        )

    def forward(self, x):
        if not self.is_layer_built:
            self.build(input_shape=x.shape)
            self.to(x.device)

        out = x
        # b, s, out_features, w / n, h / n
        out = out.mean(dim=4)
        # b, s, out_features, w / n
        out = out.mean(dim=3)
        # b, s, out_features
        out = out.mean(dim=1)
        # b, out_features

        return out


class AvgPoolFlexibleDimension(nn.Module):
    def __init__(self, dim: int):
        super(AvgPoolFlexibleDimension, self).__init__()
        self.is_layer_built = False
        self.dim = dim

    def build(self, input_shape):
        out = torch.zeros(input_shape)

        assert len(input_shape) >= self.dim, (
            "Length of shape is smaller than {}, please ensure the tensor "
            "shape is larger or equal in length".format(self.dim_to_pool_over)
        )

        out = out.mean(dim=self.dim)

        self.is_layer_built = True

        logging.debug(
            f"Build {self.__class__.__name__} with input shape {input_shape} with "
            f"output shape {out.shape}"
        )

    def forward(self, x):
        assert len(x.shape) >= self.dim, (
            "Length of shape is smaller than {}, please ensure the tensor "
            "shape is larger or equal in length".format(self.dim_to_pool_over)
        )

        if not self.is_layer_built:
            self.build(input_shape=x.shape)
            self.to(x.device)

        out = x

        out = out.mean(dim=self.dim)

        return out


class Unsqueeze(nn.Module):
    def __init__(self, dim: int):
        self.dim = dim
        super(Unsqueeze, self).__init__()

    def forward(self, x):
        return x.unsqueeze(self.dim)


class AutoResNet(ClassificationModel):
    def __init__(
        self, num_classes: int, model_name_to_download: str, pretrained: bool = True
    ):
        feature_embedding_modules = [
            ResNet,
            AvgPoolFlexibleDimension,
            AvgPoolFlexibleDimension,
        ]
        feature_embeddings_args = [
            dict(
                model_name_to_download=model_name_to_download,
                pretrained=pretrained,
            ),
            dict(dim=2),
            dict(dim=2),
        ]
        super(AutoResNet, self).__init__(
            num_classes=num_classes,
            feature_embedding_module_list=feature_embedding_modules,
            feature_embedding_args=feature_embeddings_args,
        )


class AutoConvNet(ClassificationModel):
    def __init__(
        self,
        num_classes: int,
        kernel_size: Tuple[int],
        filter_list: List[int],
        stride: Tuple[int],
        padding: int,
    ):
        feature_embedding_modules = [Conv2dEmbedding]
        feature_embeddings_args = [
            dict(
                kernel_size=kernel_size,
                layer_filter_list=filter_list,
                stride=stride,
                padding=padding,
            )
        ]
        super(AutoConvNet, self).__init__(
            num_classes=num_classes,
            feature_embedding_module_list=feature_embedding_modules,
            feature_embedding_args=feature_embeddings_args,
        )


class DenseBlock(nn.Module):
    def __init__(
        self,
        num_filters,
        dilation_factor,
        kernel_size,
        stride,
        downsample_output_size=None,
        processing_block_type=Conv2dBNLeakyReLU,
    ):
        super(DenseBlock, self).__init__()
        self.num_filters = num_filters
        self.dilation_factor = dilation_factor
        self.downsample_output_size = downsample_output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.processing_block_type = processing_block_type
        self.is_built = False

    def build(self, input_shape):
        dummy_x = torch.zeros(input_shape)

        self.conv_bn_relu_in = self.processing_block_type(
            out_channels=self.num_filters,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.dilation_factor,
            dilation=self.dilation_factor,
            bias=False,
        )

        out = self.conv_bn_relu_in.forward(dummy_x)

        if self.downsample_output_size is not None:
            if len(out.shape) == 4:
                out = F.adaptive_avg_pool2d(
                    output_size=self.downsample_output_size, input=out
                )
                dummy_x = F.adaptive_avg_pool2d(
                    output_size=self.downsample_output_size, input=dummy_x
                )
            elif len(out.shape) == 3:
                out = F.adaptive_avg_pool1d(
                    output_size=self.downsample_output_size, input=out
                )
                dummy_x = F.adaptive_avg_pool1d(
                    output_size=self.downsample_output_size, input=dummy_x
                )
            else:
                return ValueError("Dimensions of features are neither 3 nor 4")

        out = torch.cat([dummy_x, out], dim=1)

        self.is_built = True
        logging.debug(
            f"Built module {self.__class__.__name__} with input -> output sizes "
            f"{dummy_x.shape} {out.shape}"
        )

    def forward(self, x):

        if not self.is_built:
            self.build(x.shape)

        out = self.conv_bn_relu_in.forward(x)
        if self.downsample_output_size is not None:
            # logging.debug(f'Downsampling output size: {out.shape}'
            #              f'{self.downsample_output_size}')
            if len(out.shape) == 4:
                out = F.avg_pool2d(kernel_size=2, input=out)
                x = F.avg_pool2d(kernel_size=2, input=x)
            elif len(out.shape) == 3:
                # logging.debug(out.shape)
                out = F.avg_pool1d(kernel_size=2, input=out)
                x = F.avg_pool1d(kernel_size=2, input=x)
            else:
                return ValueError("Dimensions of features are neither 3 nor 4")

        out = torch.cat([x, out], dim=1)

        return out


class ConvPool1DStemBlock(nn.Module):
    def __init__(
        self,
        num_filters,
        kernel_size,
        padding,
        dilation,
        pool_kernel_size,
        pool_stride,
        **kwargs,
    ):
        super(ConvPool1DStemBlock, self).__init__()
        self.layer_dict = nn.ModuleDict()
        self.is_built = False
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride

    def build(self, input_shape):
        dummy_x = torch.zeros(input_shape)
        out = dummy_x

        self.layer_dict["stem_conv"] = nn.Conv1d(
            in_channels=out.shape[1],
            out_channels=self.num_filters,
            kernel_size=self.kernel_size,
            stride=2,
            padding=self.padding,
            bias=False,
        )

        self.layer_dict["stem_leaky_relu"] = nn.LeakyReLU(0.1)

        self.layer_dict["stem_instance_norm"] = nn.InstanceNorm1d(
            self.num_filters, affine=True, track_running_stats=True
        )

        self.layer_dict["stem_pool"] = nn.AvgPool1d(
            kernel_size=self.pool_kernel_size, stride=self.pool_stride
        )

        out = self.layer_dict["stem_conv"].forward(out)
        out = self.layer_dict["stem_leaky_relu"].forward(out)
        out = self.layer_dict["stem_instance_norm"].forward(out)

        self.is_built = True
        build_string = (
            f"Built module {self.__class__.__name__} "
            f"with input -> output sizes {dummy_x.shape} {out.shape}"
        )

        logging.info(build_string)

    def forward(self, x):
        if not self.is_built:
            self.build(x.shape)

        out = x
        out = self.layer_dict["stem_conv"].forward(out)
        out = self.layer_dict["stem_leaky_relu"].forward(out)
        out = self.layer_dict["stem_instance_norm"].forward(out)

        return out


class ResNetBlock1D(nn.Module):
    def __init__(self, num_filters, kernel_size, dilation, **kwargs):
        super(ResNetBlock1D, self).__init__()
        self.layer_dict = nn.ModuleDict()
        self.is_built = False
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dilation = dilation

    def build(self, input_shape):

        dummy_x = torch.zeros(input_shape)

        self.layer_dict["conv_filter_pool"] = nn.Conv1d(
            in_channels=dummy_x.shape[1],
            out_channels=self.num_filters,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        dummy_x = self.layer_dict["conv_filter_pool"].forward(dummy_x)

        log.debug(f"dummy_x shape: {dummy_x.shape}")

        out = dummy_x

        self.layer_dict["conv_0"] = nn.Conv1d(
            in_channels=out.shape[1],
            out_channels=self.num_filters,
            kernel_size=self.kernel_size,
            stride=self.kernel_size,
            padding=0,
            bias=False,
        )

        out = self.layer_dict["conv_0"].forward(out)

        log.debug(f"out shape: {out.shape}")

        self.layer_dict["leaky_relu_0"] = nn.LeakyReLU(0.1)

        out = self.layer_dict["leaky_relu_0"].forward(out)

        log.debug(f"out shape: {out.shape}")

        self.layer_dict["instance_norm_0"] = nn.InstanceNorm1d(
            self.num_filters, affine=True, track_running_stats=True
        )

        out = self.layer_dict["instance_norm_0"].forward(out)

        log.debug(f"out shape: {out.shape}")

        self.layer_dict["conv_spatial_pool"] = nn.Conv1d(
            in_channels=dummy_x.shape[1],
            out_channels=self.num_filters,
            kernel_size=self.kernel_size,
            stride=self.kernel_size,
            padding=0,
            bias=False,
        )

        dummy_x = self.layer_dict["conv_spatial_pool"].forward(dummy_x)

        log.debug(f"dummy_x shape: {dummy_x.shape}")

        out = dummy_x + out

        log.debug(f"out shape: {out.shape}")

        self.is_built = True

        build_string = (
            f"Built module {self.__class__.__name__} "
            f"with input -> output sizes {input_shape} {out.shape}"
        )

        logging.info(build_string)

    def forward(self, x):
        if not self.is_built:
            self.build(x.shape)

        x = self.layer_dict["conv_filter_pool"].forward(x)

        out = x

        out = self.layer_dict["conv_0"].forward(out)

        out = self.layer_dict["leaky_relu_0"].forward(out)

        out = self.layer_dict["instance_norm_0"].forward(out)

        x = self.layer_dict["conv_spatial_pool"].forward(x)

        out = x + out

        return out


class ResNetReductionBlock1D(nn.Module):
    def __init__(self, num_filters, kernel_size, dilation, reduction_rate, **kwargs):
        super(ResNetReductionBlock1D, self).__init__()
        self.layer_dict = nn.ModuleDict()
        self.is_built = False
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.reduction_rate = reduction_rate

    def build(self, input_shape):

        dummy_x = torch.zeros(input_shape)

        self.layer_dict["conv_filter_pool"] = nn.Conv1d(
            in_channels=dummy_x.shape[1],
            out_channels=self.num_filters,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        dummy_x = self.layer_dict["conv_filter_pool"].forward(dummy_x)

        self.layer_dict["conv_1"] = nn.Conv1d(
            in_channels=dummy_x.shape[1],
            out_channels=self.num_filters,
            kernel_size=self.kernel_size,
            stride=1,
            padding=0,
            bias=False,
        )

        self.layer_dict["reduction_pool"] = nn.AvgPool1d(
            kernel_size=self.reduction_rate, stride=self.reduction_rate
        )

        self.layer_dict["conv_2"] = nn.Conv1d(
            in_channels=self.num_filters,
            out_channels=self.num_filters,
            kernel_size=self.kernel_size,
            stride=1,
            padding=0,
            bias=False,
        )
        self.layer_dict["gelu_0"] = nn.LeakyReLU(0.1)
        self.layer_dict["instance_norm_0"] = nn.InstanceNorm1d(
            self.num_filters, affine=True, track_running_stats=True
        )

        self.layer_dict["gelu_1"] = nn.LeakyReLU(0.1)
        self.layer_dict["instance_norm_1"] = nn.InstanceNorm1d(
            self.num_filters, affine=True, track_running_stats=True
        )

        out = dummy_x

        pool_x = self.layer_dict["reduction_pool"](dummy_x)
        out = self.layer_dict["conv_1"].forward(out)
        out = self.layer_dict["gelu_0"].forward(out)
        out = self.layer_dict["instance_norm_0"].forward(out)
        pool_out = self.layer_dict["reduction_pool"].forward(out)
        out = self.layer_dict["conv_2"].forward(pool_out)
        out = self.layer_dict["gelu_1"].forward(out)
        out = self.layer_dict["instance_norm_1"].forward(out)

        out = pool_x + out

        self.is_built = True

        build_string = (
            f"Built module {self.__class__.__name__} "
            f"with input -> output sizes {dummy_x.shape} {out.shape}"
        )

        logging.debug(build_string)

    def forward(self, x):
        if not self.is_built:
            self.build(x.shape)

        x = self.layer_dict["conv_filter_pool"].forward(x)
        pool_x = self.layer_dict["reduction_pool"](x)
        out = x

        out = self.layer_dict["conv_1"].forward(out)
        out = self.layer_dict["gelu_0"].forward(out)
        out = self.layer_dict["instance_norm_0"].forward(out)
        pool_out = self.layer_dict["reduction_pool"].forward(out)
        out = self.layer_dict["conv_2"].forward(pool_out)
        out = self.layer_dict["gelu_1"].forward(out)
        out = self.layer_dict["instance_norm_1"].forward(out)

        out = pool_x + out

        return out


class ConvNetEmbedding(nn.Module):
    def __init__(
        self,
        num_filters: int,
        num_stages: int,
        num_blocks: int,
        kernel_size: int,
        stem_block_type: nn.Module,
        processing_block_type: nn.Module,
        reduction_block_type: nn.Module,
        dilated: bool = False,
    ):
        super(ConvNetEmbedding, self).__init__()
        self.layer_dict = nn.ModuleDict()
        self.is_built = False
        self.num_filters = num_filters
        self.num_stages = num_stages
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.stem_block_type = stem_block_type
        self.processing_block_type = processing_block_type
        self.reduction_block_type = reduction_block_type
        self.dilated = dilated

    def build(self, input_shape):
        dummy_x = torch.zeros(input_shape)
        out = dummy_x

        dilation_factor = 1

        self.layer_dict["stem_conv"] = self.stem_block_type(
            num_filters=4 * self.num_filters,
            kernel_size=self.kernel_size,
            stride=self.kernel_size,
            padding=0,
            dilation=1,
            pool_kernel_size=2,
            pool_stride=2,
        )

        out = self.layer_dict["stem_conv"].forward(out)

        for stage_idx in range(self.num_stages):

            for block_idx in range(self.num_blocks):

                if self.dilated:
                    dilation_factor = 2**block_idx

                self.layer_dict[
                    f"stage_{stage_idx}_block_{block_idx}"
                ] = self.processing_block_type(
                    num_filters=(stage_idx * self.num_filters) + self.num_filters,
                    kernel_size=self.kernel_size,
                    dilation=dilation_factor,
                )

                out = self.layer_dict[f"stage_{stage_idx}_block_{block_idx}"].forward(
                    out
                )

            self.layer_dict[
                f"stage_{stage_idx}_dim_reduction"
            ] = self.reduction_block_type(
                num_filters=(stage_idx * self.num_filters) + self.num_filters,
                kernel_size=self.kernel_size,
                dilation=dilation_factor,
                reduction_rate=2,
            )

            out = self.layer_dict[f"stage_{stage_idx}_dim_reduction"].forward(out)

        self.is_built = True
        logging.info(
            f"Built module {self.__class__.__name__} with input -> output sizes "
            f"{dummy_x.shape} {out.shape}"
        )

    def forward(self, x):

        if not self.is_built:
            self.build(input_shape=x.shape)

        out = x

        out = self.layer_dict["stem_conv"].forward(out)

        for stage_idx in range(self.num_stages):

            for block_idx in range(self.num_blocks):
                out = self.layer_dict[f"stage_{stage_idx}_block_{block_idx}"].forward(
                    out
                )

            out = self.layer_dict[f"stage_{stage_idx}_dim_reduction"].forward(out)

        return out


class AutoConv2DDenseNet(ClassificationModel):
    def __init__(
        self,
        num_classes,
        num_filters,
        num_stages,
        num_blocks,
        use_squeeze_excite_attention,
        dilated=False,
        **kwargs,
    ):
        feature_embedding_modules = [
            nn.InstanceNorm2d,
            ConvNetEmbedding,
            nn.Flatten,
        ]

        feature_embeddings_args = [
            dict(num_features=3, affine=True, track_running_stats=True),
            dict(
                num_filters=num_filters,
                num_stages=num_stages,
                num_blocks=num_blocks,
                dilated=dilated,
                processing_block_type=SqueezeExciteConv2dBNLeakyReLU
                if use_squeeze_excite_attention
                else Conv2dBNLeakyReLU,
            ),
            dict(start_dim=1, end_dim=-1),
        ]

        super(AutoConv2DDenseNet, self).__init__(
            num_classes=num_classes,
            feature_embedding_module_list=feature_embedding_modules,
            feature_embedding_args=feature_embeddings_args,
        )


class AutoConv1DDenseNet(ClassificationModel):
    def __init__(
        self,
        num_classes,
        num_filters,
        num_stages,
        num_blocks,
        use_squeeze_excite_attention,
        dilated=False,
        **kwargs,
    ):
        feature_embedding_modules = [
            nn.InstanceNorm1d,
            ConvNetEmbedding,
            nn.AdaptiveMaxPool1d,
        ]

        feature_embeddings_args = [
            dict(num_features=2, affine=True, track_running_stats=True),
            dict(
                num_filters=num_filters,
                num_stages=num_stages,
                num_blocks=num_blocks,
                dilated=dilated,
                processing_block_type=SqueezeExciteConv1dBNLeakyReLU
                if use_squeeze_excite_attention
                else Conv1dBNLeakyReLU,
            ),
            dict(output_size=2),
        ]

        super(AutoConv1DDenseNet, self).__init__(
            num_classes=num_classes,
            feature_embedding_module_list=feature_embedding_modules,
            feature_embedding_args=feature_embeddings_args,
        )


class Permute(nn.Module):
    def __init__(self, permute_order: list):
        super().__init__()
        self.permute_order = permute_order

    def forward(self, x):
        return x.permute(self.permute_order)


class AutoConv1DDenseNetVideo(ClassificationModel):
    def __init__(
        self,
        num_classes,
        num_filters,
        num_stages,
        num_blocks,
        use_squeeze_excite_attention,
        dilated=False,
        **kwargs,
    ):
        feature_embedding_modules = [Permute, ConvNetEmbedding, nn.Flatten]

        feature_embeddings_args = [
            dict(permute_order=[0, 2, 1]),
            dict(
                num_filters=num_filters,
                num_stages=num_stages,
                num_blocks=num_blocks,
                dilated=dilated,
                processing_block_type=SqueezeExciteConv1dBNLeakyReLU
                if use_squeeze_excite_attention
                else Conv1dBNLeakyReLU,
            ),
            dict(start_dim=1, end_dim=-1),
        ]

        super(AutoConv1DDenseNetVideo, self).__init__(
            num_classes=num_classes,
            feature_embedding_module_list=feature_embedding_modules,
            feature_embedding_args=feature_embeddings_args,
        )


class AutoTextNet(ClassificationModel):
    def __init__(
        self,
        vocab_size: int,
        num_filters: int,
        num_stages: int,
        num_blocks: int,
        use_squeeze_excite_attention: bool,
        dilated: bool,
        num_classes: int,
        **kwargs,
    ):
        feature_embedding_modules = [
            nn.Embedding,
            ConvNetEmbedding,
            nn.Flatten,
        ]

        feature_embeddings_args = [
            dict(num_embeddings=vocab_size, embedding_dim=num_filters),
            dict(
                num_filters=num_filters,
                num_stages=num_stages,
                num_blocks=num_blocks,
                dilated=dilated,
                processing_block_type=SqueezeExciteConv1dBNLeakyReLU
                if use_squeeze_excite_attention
                else Conv1dBNLeakyReLU,
            ),
            dict(),
        ]

        super(AutoTextNet, self).__init__(
            num_classes=num_classes,
            feature_embedding_module_list=feature_embedding_modules,
            feature_embedding_args=feature_embeddings_args,
            input_type=torch.long,
        )
