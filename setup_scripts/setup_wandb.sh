# This is secret and shouldn't be checked into version control
export WANDB_API_KEY=$MY_WANDB_API_KEY
# Name and notes optional
#WANDB_NAME="My first run"
#WANDB_NOTES="Smaller learning rate, more regularization."

# Only needed if you don't checkin the wandb/settings file
export WANDB_ENTITY=machinelearningbrewery
export WANDB_PROJECT=TALI-experiments-v-4-0
# export WANDB_DIR=$EXPERIMENTS_DIR
# If you don't want your script to sync to the cloud
# os.environ['WANDB_MODE'] = 'offline'