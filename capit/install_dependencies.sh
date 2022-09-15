#!/bin/bash
# Install CAPIT dependencies
mamba install git -y
mamba install opencv -y
mamba install h5py -y
mamba install pytorch-lightning -y
mamba install transformers -y
mamba install GPUtil -y
mamba install orjson -y
mamba install tqdm -y
mamba install regex -y
mamba install pudb -y
mamba install seaborn -y
mamba install scikit-learn -y
mamba install pytest -y
mamba install rich -y
mamba install python-dotenv -y
mamba install flake8 -y
mamba install isort -y
mamba install black -y
mamba install pre-commit -y
mamba install wandb -y
mamba install hydra-core -y
mamba install google-cloud-storage -y
mamba install google-api-python-client -y
mamba install cryptography -y
mamba install ftfy -y
mamba install imutils -y
mamba install scipy -y
mamba install einops -y
mamba install torchmetrics -y
mamba install ffmpeg -y
mamba install tensorflow tensorflow-datasets -y
echo yes | pip install hub timm jsonlint nvidia-ml-py3 testresources hydra hydra-core hydra-colorlog hub hydra-optuna-sweeper dotted_dict ray higher --upgrade
echo yes | pip install git+https://github.com/openai/CLIP.git@main
echo yes | pip install git+https://github.com/AntreasAntoniou/TALI.git@main
echo yes | pip install learn2learn

echo yes | pip install -e .