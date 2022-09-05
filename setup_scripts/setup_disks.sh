########################################################################################
export HOME="/root/"
export MOUNT_TRAIN_DIR="/mnt/disk/tali-train"
export MOUNT_EVAL_DIR="/mnt/disk/tali-eval"
########################################################################################
export EXPERIMENTS_DIR="$HOME/experiments"
export EXPERIMENT_DIR="$HOME/experiments"
export PATH_CACHE_DIR="$HOME/path_cache"
########################################################################################
export TRAIN_DATASET_DIR="$MOUNT_TRAIN_DIR/dataset/train"
export VAL_DATASET_DIR="$MOUNT_EVAL_DIR/dataset/val"
export TEST_DATASET_DIR="$MOUNT_EVAL_DIR/dataset/test"
########################################################################################
if [ ! -d "$PATH_CACHE_DIR" ]; then
  mkdir -p $PATH_CACHE_DIR
  chmod -Rv 777 $PATH_CACHE_DIR
fi
########################################################################################
if [ ! -d "$EXPERIMENTS_DIR" ]; then
  mkdir -p $EXPERIMENTS_DIR
  chmod -Rv 777 $EXPERIMENTS_DIR
fi
########################################################################################
if [ ! -d "$TRAIN_DATASET_DIR" ]; then
  mkdir -p $TRAIN_DATASET_DIR
  chmod -Rv 777 $TRAIN_DATASET_DIR
fi

if [ ! -d "$VAL_DATASET_DIR" ]; then
  mkdir -p $VAL_DATASET_DIR
  chmod -Rv 777 $VAL_DATASET_DIR
fi

if [ ! -d "$TEST_DATASET_DIR" ]; then
  mkdir -p $TEST_DATASET_DIR
  chmod -Rv 777 $TEST_DATASET_DIR
fi
########################################################################################

#mount -o discard,defaults /dev/sdb $MOUNT_DIR
mount -o ro,noload /dev/sdc $MOUNT_TRAIN_DIR
mount -o ro,noload /dev/sdb $MOUNT_EVAL_DIR

########################################################################################




