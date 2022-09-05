#!/bin/bash
echo "Running admin defined startup script"
#export WANDB_SWEEP_ID=tyhuzmes
export CODE_DIR=/root/target_codebase

rm -rf $CODE_DIR
git clone https://github.com/AntreasAntoniou/TALI-lightning-hydra.git $CODE_DIR

echo "Launching experiment"
#export RUNNER_SCRIPT=$CODE_DIR/experiment_runner_scripts/base_base_modus_prime_resnet50_image=True_audio=False_video=False_text=True_auto_scale_batch_size=True.sh
screen -dmS startup_script_session bash -c 'bash $RUNNER_SCRIPT; exec bash'


# Path: setup_scripts/startup_script_template.sh