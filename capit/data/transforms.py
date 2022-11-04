import torch.nn as nn
from torchvision import transforms


class ToThreeChannels(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image):

        if image.shape[0] == 1:
            return image.repeat(3, 1, 1)
        elif image.shape[0] == 4:
            return image[:3]
        else:
            return image
