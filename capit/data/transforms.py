from torchvision import transforms
import torch.nn as nn
from capit.data.tokenizers import HuggingFaceBPETokenizer

tokenizer = HuggingFaceBPETokenizer(context_length=77)


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


def image_transforms_base():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            ToThreeChannels(),
        ]
    )


def text_transforms_base():
    return transforms.Compose(
        [
            lambda x: tokenizer.forward(x),
        ]
    )
