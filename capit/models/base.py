from __future__ import print_function

import logging

import torch
import torch.nn as nn


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
