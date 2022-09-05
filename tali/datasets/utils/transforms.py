import logging

import torch

log = logging.getLogger(__name__)


class SubSampleVideoFrames(torch.nn.Module):
    """Crops the given image at the center.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
    """

    def __init__(self, num_frames):
        self.num_frames = num_frames
        super().__init__()

    def forward(self, sequence_of_frames):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        total_num_frames = sequence_of_frames.shape[0]
        maximum_start_point = total_num_frames - self.num_frames

        if maximum_start_point < 0:
            padding = torch.zeros(
                size=(
                    -maximum_start_point,
                    sequence_of_frames.shape[1],
                    sequence_of_frames.shape[2],
                    sequence_of_frames.shape[3],
                )
            )
            sequence_of_frames = torch.cat([sequence_of_frames, padding], dim=0)

        else:
            choose_start_point = torch.randint(
                low=0, high=maximum_start_point + 1, size=(1,)
            )[0]

            sequence_of_frames = sequence_of_frames[
                choose_start_point : choose_start_point + self.num_frames
            ]

        # log.debug(sequence_of_frames.shape)

        return sequence_of_frames

    def __repr__(self):
        return self.__class__.__name__ + "(size={0})".format(self.size)


class SubSampleAudioFrames(torch.nn.Module):
    """Crops the given image at the center.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
    """

    def __init__(self, num_frames):
        self.num_frames = num_frames
        super().__init__()

    def forward(self, sequence_of_audio_frames):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        # logging.info(f"{sequence_of_audio_frames.shape} {self.num_frames}")
        total_num_frames = sequence_of_audio_frames.shape[1]

        if self.num_frames <= total_num_frames:
            return sequence_of_audio_frames[:, 0 : self.num_frames]

        padding_size = self.num_frames - total_num_frames

        sequence_of_audio_frames = torch.cat(
            [
                sequence_of_audio_frames,
                torch.zeros(sequence_of_audio_frames.shape[0], padding_size),
            ],
            dim=1,
        )
        return sequence_of_audio_frames

    def __repr__(self):
        return self.__class__.__name__ + "(size={0})".format(self.size)
