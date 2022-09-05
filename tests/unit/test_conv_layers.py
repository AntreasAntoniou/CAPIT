import pytest
import torch

from tali.models.auto_builder.conv_modules import (
    ConvPool1DStemBlock,
    ResNetBlock1D,
    ResNetReductionBlock1D,
    ConvNetEmbedding,
)


@pytest.mark.parametrize("num_filters", [32, 128, 256, 1024])
@pytest.mark.parametrize("kernel_size", [3])
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("padding", [1])
@pytest.mark.parametrize("dilation", [1])
@pytest.mark.parametrize("pool_kernel_size", [2])
@pytest.mark.parametrize("pool_stride", [2])
def test_stem_module(
    num_filters, kernel_size, stride, padding, dilation, pool_kernel_size, pool_stride
):
    module = ConvPool1DStemBlock(
        num_filters,
        kernel_size,
        stride,
        padding,
        dilation,
        pool_kernel_size,
        pool_stride,
    )

    x = torch.randn(2, 10, 100)
    out = module.forward(x)
    print(out.shape)
    assert out.shape == (2, num_filters, 50)


@pytest.mark.parametrize("num_filters", [32, 128, 256, 512, 1024, 2048])
@pytest.mark.parametrize("kernel_size", [3])
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("padding", [1])
def test_processing_module(num_filters, kernel_size, stride, padding):
    module = ResNetBlock1D(
        num_filters,
        kernel_size,
        stride,
        padding,
    )

    x = torch.randn(2, num_filters, 100)
    out = module.forward(x)
    print(out.shape)
    assert len(out.shape) == 3


@pytest.mark.parametrize("num_filters", [32, 128, 256, 512, 1024, 2048])
@pytest.mark.parametrize("kernel_size", [3])
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("padding", [1])
@pytest.mark.parametrize("reduction_rate", [1])
def test_reduction_module(num_filters, kernel_size, stride, padding, reduction_rate):
    module = ResNetReductionBlock1D(
        num_filters, kernel_size, stride, padding, reduction_rate
    )

    x = torch.randn(2, num_filters, 100)
    out = module.forward(x)
    print(out.shape)
    assert len(out.shape) == 3


@pytest.mark.parametrize("num_filters", [32])
@pytest.mark.parametrize("num_stages", [3])
@pytest.mark.parametrize("num_blocks", [4])
@pytest.mark.parametrize("kernel_size", [3])
def test_resnet_module(num_filters, num_stages, num_blocks, kernel_size):
    module = ConvNetEmbedding(
        num_filters=num_filters,
        num_stages=num_stages,
        num_blocks=num_blocks,
        kernel_size=kernel_size,
        stem_block_type=ConvPool1DStemBlock,
        processing_block_type=ResNetBlock1D,
        reduction_block_type=ResNetReductionBlock1D,
    )

    x = torch.randn(2, 2, 441000 // 2)
    out = module.forward(x)
    print(out.shape)
    assert len(out.shape) == 3
