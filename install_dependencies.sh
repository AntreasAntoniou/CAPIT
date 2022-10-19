#!/bin/bash
# Install CAPIT dependencies
echo yes | pip install hydra_zen
echo yes | pip install git+https://github.com/AntreasAntoniou/wandb_stateless_utils.git
echo yes | pip install -e .