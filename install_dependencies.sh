#!/bin/bash
# Install CAPIT dependencies
mamba install git -y
mamba install opencv -y
mamba install h5py -y
mamba install pytorch-lightning -y
mamba install transformers -y
mamba install orjson -y
mamba install tqdm -y
mamba install regex -y
mamba install seaborn -y
mamba install scikit-learn -y
mamba install rich -y
mamba install python-dotenv -y
mamba install ftfy -y
mamba install imutils -y
mamba install scipy -y
mamba install einops -y
mamba install torchmetrics -y
mamba install ffmpeg -y
mamba install tensorflow tensorflow-datasets -y
echo yes | pip install wandb opencv-contrib-python hub timm hydra hydra-core dotted_dict higher --upgrade
echo yes | pip install git+https://github.com/openai/CLIP.git@main
echo yes | pip install git+https://github.com/AntreasAntoniou/TALI.git@main
echo yes | pip install learn2learn

echo yes | pip install -e .