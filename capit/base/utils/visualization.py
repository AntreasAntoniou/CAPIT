import torch
import torchvision


def image_batch_to_grid(
    batch: torch.Tensor,
    nrow: int = 10,
    padding: int = 0,
    normalize: bool = False,
    range: tuple = None,
    scale_each: bool = False,
    pad_value: float = 0,
) -> torch.Tensor:
    grid = torchvision.utils.make_grid(
        batch,
        nrow=nrow,
        padding=padding,
        normalize=normalize,
        range=range,
        scale_each=scale_each,
        pad_value=pad_value,
    )
    return grid
