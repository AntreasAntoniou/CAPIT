#!/usr/bin/env python

"""
Builds a convolutional neural network on the fashion mnist data set.

Designed to show wandb integration with pytorch.
"""

import os
import random
from time import sleep

import GPUtil
import numpy as np
import wandb

from tali.base import utils

log = utils.get_logger(__name__)

# write remote script that interfaces with wandb to launch a machine with specific GPUs
# and run single experiment, then kill machine


# single_gpu_config=[dict(use_image_modality=True,
#                         use_audio_modality=[True, False],
#                         use_text_modality=[True, False],
#                         use_video_modality=False)]


hyperparameter_defaults = dict(
    use_image_modality=True,
    use_video_modality=False,
    use_audio_modality=False,
    use_text_modality=True,
    datamodule_name="base",
    model_name="base_modus_prime_resnet50",
    batch_size=64,
)

wandb.init(config=hyperparameter_defaults, project="TALI-gcp-sweep-1")
config = wandb.config


def main():

    if config.model_name in [
        "base_modus_prime_resnet50",
        "base_modus_prime_vi-transformer16",
    ]:
        score = np.sum(
            np.array([config.use_text_modality, config.use_audio_modality]).astype(
                np.int32
            )
        )

        num_gpus = 8 if config.use_video_modality else 2 * score

    elif config.model_name in [
        "centi_modus_prime_resnet50",
        "centi_modus_prime_vi-transformer16",
    ]:
        score = np.sum(
            np.array([config.use_text_modality, config.use_audio_modality]).astype(
                np.int32
            )
        )

        num_gpus = 2 if config.use_video_modality else 1
    else:
        raise NotImplementedError(
            f"Given config does not fall into " f"the expected patterns {config}"
        )

    deviceIDs = []

    while len(deviceIDs) < num_gpus:
        log.info(
            f"Need {num_gpus} GPUs, but have access to {len(deviceIDs)} 🏋🖥\n"
            f"Waiting for GPUs to become available..🐈🪑"
        )
        sleep(random.randint(0, 60))
        deviceIDs = GPUtil.getAvailable(
            order="first",
            limit=8,
            maxLoad=0.001,
            maxMemory=0.001,
            includeNan=False,
            excludeID=[],
            excludeUUID=[],
        )

    template_command = (
        # f"fuser -k /dev/nvidia*; "
        f"python $CODE_DIR/run.py hydra.verbose=True trainer=default "
        f"resume=True batch_size={config.batch_size} "
        f"wandb_project_name=TALI-gcp-sweep-1 "
        f"trainer.gpus={num_gpus} "
        f"trainer.auto_scale_batch_size=False "
        f"datamodule.dataset_config.rescan_paths=True datamodule.prefetch_factor=3 "
        f"datamodule.num_workers={int(num_gpus * 12)} "
        f"model={config.model_name} "
        f"datamodule.dataset_config.dataset_size_identifier={config.datamodule_name} "
        f"datamodule.dataset_config.modality_config.image={config.use_image_modality} "
        f"datamodule.dataset_config.modality_config.text={config.use_text_modality} "
        f"datamodule.dataset_config.modality_config.audio={config.use_audio_modality} "
        f"datamodule.dataset_config.modality_config.video={config.use_video_modality}\n\n"
    )
    log.info(template_command)
    os.system(template_command)


if __name__ == "__main__":
    main()
