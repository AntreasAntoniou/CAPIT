from dataclasses import dataclass

DATASET_DIR = "${data_dir}"
SEED = "${seed}"
ROOT_EXPERIMENT_DIR = "${root_experiment_dir}"
CURRENT_EXPERIMENT_DIR = "${current_experiment_dir}"
CODE_DIR = "${code_dir}"
EXPERIMENT_NAME = "${name}"
BATCH_SIZE = "${batch_size}"
CHECKPOINT_DIR = "${current_experiment_dir}/checkpoints/"


@dataclass
class MaxDurationTypes:
    MAX_EPOCHS: str = "${trainer.max_epochs}"
    MAX_STEPS: str = "${trainer.max_steps}"
