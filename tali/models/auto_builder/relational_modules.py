import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def check_spatial_size_maybe_avg_pool(x, avg_pool_output_size):
    """
    Receives a tensor x, and a kernel size and checks spatial dimensions, potentially avg pooling if the size is not as expected
    :param x: A tensor
    :param avg_pool_output_size: An integer defining the 'oatch size' to be used for avg pooling
     to ensure the output tensor has the right dimensionality (that is to have an input
     of spatial dimensions 'kernel size'
    :return: A tensor with spatial size 'kernel size'
    """
    out = x

    if len(x.shape) == 4:
        b, c, h, w = out.shape
        if h != avg_pool_output_size:
            out = F.adaptive_avg_pool2d(out, output_size=avg_pool_output_size)
            b, c, h, w = out.shape

        out = out.view(b, c, h * w)

    elif len(x.shape) == 3:
        b, c, h = out.shape
        if h != avg_pool_output_size:
            out = F.adaptive_avg_pool1d(out, output_size=avg_pool_output_size)
            b, c, h = out.shape

        out = out.view(b, c, h)
    else:
        raise ValueError(
            f"Dimensions of x must be either 3 or 4 dimensions, "
            f"instead the shape is {x.shape}, and thus {len(x.shape)} dimensions"
        )

    return out


def generate_spatial_coordinate_tensor(x, spatial_length):
    """
    Given
    :param x:
    :param spatial_length:
    :return:
    """
    coord_tensor_x = (
        torch.arange(start=0, end=np.sqrt(spatial_length))
        .unsqueeze(0)
        .repeat([int(np.sqrt(spatial_length)), 1])
        .unsqueeze(2)
    )  # n, n
    coord_tensor_y = (
        torch.arange(start=0, end=np.sqrt(spatial_length))
        .unsqueeze(1)
        .repeat([1, int(np.sqrt(spatial_length))])
        .unsqueeze(2)
    )  # n, n

    coord_tensor = torch.cat([coord_tensor_x, coord_tensor_y], dim=2).view(-1, 2)

    coord_tensor = coord_tensor.unsqueeze(0) - int(np.sqrt(spatial_length) / 2)

    if coord_tensor.shape[0] != x.shape[0]:
        coord_tensor = coord_tensor[0].unsqueeze(0).repeat([x.shape[0], 1, 1])

    return coord_tensor


def generate_pair_tensor(x, spatial_length):
    x_i = torch.unsqueeze(x, 1)  # (1xh*wxc)
    x_i = x_i.repeat(1, spatial_length, 1, 1)  # (h*wxh*wxc)
    x_j = torch.unsqueeze(x, 2)  # (h*wx1xc)
    x_j = x_j.repeat(1, 1, spatial_length, 1)  # (h*wxh*wxc)

    return torch.cat([x_i, x_j], 3)


def collect_new_indices_for_given_dimensionality(
    x_img,
    relational_patch_size,
    avg_pool_patch_size,
    coord_tensor,
    patch_coordinate_tensor,
):
    out_img = check_spatial_size_maybe_avg_pool(x_img, avg_pool_patch_size)
    out_img = out_img.permute([0, 2, 1])  # b, h*w, c
    b, spatial_length, c = out_img.shape

    out_img = out_img.view(
        b, int(np.sqrt(spatial_length)), int(np.sqrt(spatial_length)), c
    )

    out_img = out_img.permute([0, 3, 1, 2])

    patch_coordinates = image_to_patch_coordinates(
        out_img, patch_size=relational_patch_size, stride=1, dilation=1
    )

    out_img = image_to_vectors(out_img, indexes_of_patches=patch_coordinates)
    out_img = out_img.permute([0, 2, 1])
    b, spatial_length, c = out_img.shape

    coord_tensor[x_img.shape] = generate_spatial_coordinate_tensor(
        x=out_img, spatial_length=spatial_length
    ).float()

    coord_tensor[x_img.shape] = coord_tensor[x_img.shape].to(out_img.device)
    patch_coordinate_tensor[x_img.shape] = patch_coordinates.to(out_img.device)

    return coord_tensor, patch_coordinate_tensor


def image_to_patch_coordinates(input_image, patch_size, stride, dilation):
    patches = []

    for i in range(0, input_image.shape[2] - (patch_size - 1), stride):
        for j in range(0, input_image.shape[3] - (patch_size - 1), stride):
            x_range = patch_size + ((patch_size - 1) * (dilation - 1))
            y_range = patch_size + ((patch_size - 1) * (dilation - 1))
            if (
                i + x_range <= input_image.shape[2]
                and j + y_range <= input_image.shape[3]
            ):
                coord = torch.Tensor(
                    [
                        x * input_image.shape[2] + y
                        for x in range(i, i + x_range, dilation)
                        for y in range(j, j + y_range, dilation)
                    ]
                )

                patches.append(coord)

    if patch_size > input_image.shape[2]:
        raise ValueError(
            f"Patch size {patch_size} is larger than the spatial dimension of the input tensor {input_image.shape}, and as a result there are 0 patches to be processed, please increase input spatial dimensions or reduce patch size"
        )

    return torch.stack(patches, dim=0).long()


def image_to_vectors(input_image, indexes_of_patches):
    p, num_pixels = indexes_of_patches.shape
    indexes_of_patches = indexes_of_patches.view(p * num_pixels)

    out = input_image.view(input_image.shape[0], input_image.shape[1], -1)

    out = torch.index_select(
        out, dim=2, index=indexes_of_patches.to(input_image.device)
    )
    out = out.view(out.shape[0], out.shape[1], p, num_pixels)

    out = out.permute([0, 1, 3, 2])

    out = out.reshape(out.shape[0], -1, out.shape[-1])

    return out


class PatchBatchRelationalModule(nn.Module):
    def __init__(
        self,
        num_layers,
        num_hidden_filters,
        num_output_channels,
        avg_pool_output_size,
        patch_size,
        use_coordinates=True,
        bias=True,
        **kwargs,
    ):
        super(PatchBatchRelationalModule, self).__init__()

        self.block_dict = nn.ModuleDict()
        self.is_built = False
        self.use_coordinates = use_coordinates
        self.num_layers = num_layers
        self.num_output_channels = num_output_channels
        self.num_hidden_filters = num_hidden_filters
        self.avg_pool_output_size = avg_pool_output_size
        self.patch_size = patch_size
        self.bias = bias
        self.coordinate_sets = {}
        self.coord_tensor = {}
        self.patch_coordinates = {}

        for key, value in kwargs.items():
            setattr(self, key, value)

    def build_block(self, x):
        out_img = x
        """g"""
        out_img = check_spatial_size_maybe_avg_pool(
            out_img, avg_pool_output_size=self.avg_pool_output_size
        )

        out_img = out_img.permute([0, 2, 1])  # b, h*w, c
        b, spatial_length, c = out_img.shape

        out_img = out_img.view(
            b, int(np.sqrt(spatial_length)), int(np.sqrt(spatial_length)), c
        )

        out_img = out_img.permute([0, 3, 1, 2])

        self.patch_coordinates[out_img.shape] = image_to_patch_coordinates(
            out_img, patch_size=self.patch_size, stride=1, dilation=1
        )

        out_img = image_to_vectors(
            out_img, indexes_of_patches=self.patch_coordinates[out_img.shape]
        )
        out_img = out_img.permute([0, 2, 1])
        b, spatial_length, c = out_img.shape

        if self.use_coordinates:
            self.coord_tensor[x.shape] = generate_spatial_coordinate_tensor(
                x=out_img, spatial_length=spatial_length
            ).float()

            self.coord_tensor[x.shape] = self.coord_tensor[x.shape].to(x.device)

            out_img = torch.cat([out_img, self.coord_tensor[x.shape]], dim=2)

        pair_features = generate_pair_tensor(x=out_img, spatial_length=spatial_length)

        pair_features = pair_features.view(
            pair_features.shape[0],
            pair_features.shape[1] * pair_features.shape[2],
            pair_features.shape[3],
        )

        image_batch, pair_batch, feature_length = pair_features.shape

        out = pair_features.view(image_batch * pair_batch, feature_length)

        for idx_layer in range(self.num_layers):
            self.block_dict["g_fcc_{}".format(idx_layer)] = nn.Linear(
                out.shape[1],
                out_features=self.num_hidden_filters,
                bias=self.bias,
            )
            self.block_dict["g_fcc_{}".format(idx_layer)].to(out.device)
            out = self.block_dict["g_fcc_{}".format(idx_layer)].forward(out)
            out = F.leaky_relu(out)

        # reshape again and sum

        out = out.view(image_batch, pair_batch, -1)
        out = out.mean(1)
        """f"""
        self.output_layer = nn.Linear(
            in_features=out.shape[1],
            out_features=self.num_output_channels,
            bias=self.bias,
        )
        self.output_layer.to(out.device)
        out = self.output_layer.forward(out)

        logging.debug(
            f"Built {self.__class__.__name__} with output volume shape " f"{out.shape}"
        )

    def forward(self, x_img):

        if not self.is_built:
            self.build_dimensionality = x_img.shape
            self.build_block(x_img)
            self.is_built = True

        if not (
            x_img.shape[0] == self.build_dimensionality[0]
            and x_img.shape[-1] == self.build_dimensionality[-1]
        ):
            self.build_dimensionality = x_img.shape
            (
                self.coord_tensor,
                self.patch_coordinates,
            ) = collect_new_indices_for_given_dimensionality(
                x_img,
                relational_patch_size=self.patch_size,
                avg_pool_patch_size=self.avg_pool_output_size,
                coord_tensor=self.coord_tensor,
                patch_coordinate_tensor=self.patch_coordinates,
            )

        out_img = x_img
        """g"""
        out_img = check_spatial_size_maybe_avg_pool(
            out_img, avg_pool_output_size=self.avg_pool_output_size
        )
        out_img = out_img.permute([0, 2, 1])  # b, h*w, c
        b, spatial_length, c = out_img.shape

        out_img = out_img.view(
            b, int(np.sqrt(spatial_length)), int(np.sqrt(spatial_length)), c
        )

        out_img = out_img.permute([0, 3, 1, 2])

        out_img = image_to_vectors(
            out_img, indexes_of_patches=self.patch_coordinates[out_img.shape]
        )
        out_img = out_img.permute([0, 2, 1])
        b, spatial_length, c = out_img.shape

        if self.use_coordinates:
            self.coord_tensor[x_img.shape] = self.coord_tensor[x_img.shape].to(
                out_img.device
            )

            out_img = torch.cat([out_img, self.coord_tensor[x_img.shape]], dim=2)

        pair_features = generate_pair_tensor(x=out_img, spatial_length=spatial_length)

        pair_features = pair_features.view(
            pair_features.shape[0],
            pair_features.shape[1] * pair_features.shape[2],
            pair_features.shape[3],
        )

        image_batch, pair_batch, feature_length = pair_features.shape

        out = pair_features.view(image_batch * pair_batch, feature_length)

        # logging.debug(out.shape, out.device)

        for idx_layer in range(self.num_layers):
            out = self.block_dict["g_fcc_{}".format(idx_layer)].forward(out)
            out = F.leaky_relu(out)

        # reshape again and sum

        out = out.view(image_batch, pair_batch, -1)
        out = out.mean(1)
        """f"""
        out = self.output_layer.forward(out)

        return out


def test_patch_relational_module():
    input_shape_list = [
        (2, 3, 32, 32),
        (10, 3, 32, 32),
        (2, 3, 32),
        (2, 32, 32),
    ]
    use_coord_list = [False, True]
    patch_size_list = [2, 4]

    for input_shape in input_shape_list:
        for use_coord in use_coord_list:
            for patch_size in patch_size_list:
                logging.debug(f"Testing {input_shape, use_coord, patch_size}")
                x_dummy = torch.zeros(input_shape)
                model = PatchBatchRelationalModule(
                    num_layers=3,
                    num_hidden_filters=32,
                    num_output_channels=64,
                    avg_pool_output_size=16,
                    patch_size=patch_size,
                    use_coordinates=use_coord,
                )
                out = model.forward(x_dummy)
