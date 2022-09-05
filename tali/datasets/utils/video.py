import logging

import cv2
import torch

log = logging.getLogger(__name__)


def load_frames(
    selected_frame_list,
    image_height,
    image_width,
    image_channels,
):
    image_tensor = torch.zeros(
        (len(selected_frame_list), image_channels, image_height, image_width)
    )
    # log.info(f"Loading frame {selected_frame_list}")
    for idx, frame_filepath in enumerate(selected_frame_list):
        try:
            image = cv2.imread(frame_filepath)
            image = (
                cv2.resize(
                    image,
                    (image_width, image_height),
                    interpolation=cv2.INTER_CUBIC,
                )
                / 255.0
            )
            image = torch.Tensor(image).permute([2, 0, 1])
            image_tensor[idx] = image

        except Exception as e:
            log.debug(
                f"Could not load image {frame_filepath} with error {e}. Skipping."
            )
            return None
    return image_tensor
