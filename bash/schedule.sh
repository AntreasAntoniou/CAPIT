#!/bin/bash
# Shedule execution of many runs
# Run from root folder with: bash bash/schedule.sh

python run.py hydra.verbose=True trainer=default \
resume=True batch_size=48 trainer.gpus=1 model=milli_modus_prime_vi-transformer16

python run.py hydra.verbose=True trainer=default \
resume=True batch_size=48 trainer.gpus=1 model=centi_modus_prime_vi-transformer16

python run.py hydra.verbose=True trainer=default \
resume=True batch_size=48 trainer.gpus=2 model=deci_modus_prime_vi-transformer16

python run.py hydra.verbose=True trainer=default \
resume=True batch_size=48 trainer.gpus=2 model=base_modus_prime_vi-transformer16

#######################################################################################

python run.py hydra.verbose=True trainer=default \
resume=True batch_size=48 trainer.gpus=1 model=milli_modus_prime_resnet50

python run.py hydra.verbose=True trainer=default \
resume=True batch_size=48 trainer.gpus=1 model=centi_modus_prime_resnet50

python run.py hydra.verbose=True trainer=default \
resume=True batch_size=48 trainer.gpus=1 model=deci_modus_prime_resnet50

python run.py hydra.verbose=True trainer=default \
resume=True batch_size=48 trainer.gpus=1 model=base-deci-hybrid_modus_prime_resnet50

python run.py hydra.verbose=True trainer=default \
resume=True batch_size=48 trainer.gpus=1 model=base_modus_prime_resnet50

#######################################################################################

datamodule.dataset_config.modality_config.image=True
datamodule.dataset_config.modality_config.text=False
datamodule.dataset_config.modality_config.video=False
datamodule.dataset_config.modality_config.audio=True