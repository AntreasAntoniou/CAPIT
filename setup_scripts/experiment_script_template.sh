#!/bin/bash
export HOME=/root/
source $HOME/.bashrc
source $HOME/conda/bin/activate
conda activate tali

cd $CODE_DIR
git pull
pip install -r $CODE_DIR/requirements.txt

source $CODE_DIR/setup_scripts/setup_base_experiment_disk.sh
source $CODE_DIR/setup_scripts/setup_wandb_credentials.sh

cd $CODE_DIR

