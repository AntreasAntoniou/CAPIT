from __future__ import print_function

import logging
from collections import OrderedDict

import torch
import torch.nn as nn

from transformers.modeling_outputs import BaseModelOutputWithPooling


class BaseLinearOutputModel(nn.Module):
    def __init__(
        self,
        feature_embedding_module_list,
        feature_embedding_args,
        embedding_output_features,
        input_type="float32",
    ):
        self.feature_embedding_module_list = feature_embedding_module_list
        self.feature_embedding_args = feature_embedding_args
        self.embedding_output_features = embedding_output_features
        self.is_layer_built = False
        self.input_type = getattr(torch, input_type)
        super(BaseLinearOutputModel, self).__init__()

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
        # logging.debug(f'{type(out)}')

        if isinstance(out, tuple):
            out = out[-1]

        if isinstance(out, BaseModelOutputWithPooling):
            out = out[1]

        out = out.view(out.shape[0], -1)
        logging.debug(f"build input {out.shape}")

        # classification head
        logging.debug(f"{out.shape[1]} {self.embedding_output_features}")
        self.output_layer = nn.Linear(
            in_features=out.shape[1],
            out_features=self.embedding_output_features,
            bias=True,
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
        :return: preds (b, embedding_output_features)
        """
        x = x.type(self.input_type)
        if not self.is_layer_built:
            self.build(input_shape=x.shape)

        self.to(x.device)

        out = x
        features = self.feature_embedding_module.forward(out)

        if isinstance(features, tuple):
            features = features[-1]

        if isinstance(features, BaseModelOutputWithPooling):
            features = features[1]

        out = features.view(features.shape[0], -1)
        out = self.output_layer.forward(out)
        return out, features


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


class ConcatenateLayer(nn.Module):
    def __init__(self):
        super(ConcatenateLayer, self).__init__()

    def forward(self, x: list, dim: int):
        x = torch.cat(tensors=x, dim=dim)
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
    def __init__(self, dim):
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
    def __init__(self, dim):
        self.dim = dim
        super(Unsqueeze, self).__init__()

    def forward(self, x):
        return x.unsqueeze(self.dim)


class Permute(nn.Module):
    def __init__(self, permute_order: list):
        super().__init__()
        self.permute_order = permute_order

    def forward(self, x):
        return x.permute(self.permute_order)
