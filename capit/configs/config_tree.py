import os
from dataclasses import MISSING, dataclass, field
from typing import Any, List, Optional

from capit.base.utils import get_logger
from capit.configs.callbacks import (
    LearningRateMonitor,
    LogConfigInformation,
    LogGrads,
    ModelSummaryConfig,
    RichProgressBar,
    SaveCheckpointsWandb,
    UploadCodeAsArtifact,
    model_checkpoint_eval,
    model_checkpoint_train,
)
from omegaconf import OmegaConf

log = get_logger(__name__)

defaults = [
    {"callbacks": "wandb"},
    {"logger": "wandb"},
    {"model": "clip-image-text"},
    {"datamodule": "InstagramImageTextMultiModal"},
    {"optimizer": "AdamW"},
    {"trainer": "gpu-dp"},
    {"mode": "base"},
    {"hydra": "custom_logging_path"},
]

overrides = []


def to_str(x):
    if isinstance(x, str):
        return x
    return str(x)


OmegaConf.register_new_resolver("last_bit", lambda x: x.split(".")[-1])
OmegaConf.register_new_resolver("lower", lambda x: x.lower())
OmegaConf.register_new_resolver("remove_slashes", lambda x: x.replace("/", "-"))
OmegaConf.register_new_resolver(
    "remove_redundant_words",
    lambda x: x.replace("scheme", "").replace("module", "").replace("config", ""),
)
OmegaConf.register_new_resolver("to_str", to_str)


def get_last_bit(x: str) -> str:
    return "${last_bit:" + str(x) + "}"


def get_lower(x: str) -> str:
    return "${lower:" + str(x) + "}"


def get_remove_slashes(x: str) -> str:
    return "${remove_slashes:" + str(x) + "}"


def get_remove_redundant_words(x: str) -> str:
    return "${remove_redundant_words:" + str(x) + "}"


def get_str(x: Any) -> str:
    return "${to_str:" + str(x) + "}"


def generate_name(
    prefix, optimizer, dataset_name, model_name, pretrained, fine_tune, seed
) -> str:
    process_string_fn = lambda x: get_remove_redundant_words(
        get_lower(get_last_bit(get_remove_slashes(get_str(x))))
    )
    name = f"{process_string_fn(prefix)}_"
    name += f"{process_string_fn(optimizer)}_"
    name += f"{process_string_fn(dataset_name)}_"
    name += f"{process_string_fn(model_name)}_"
    name += f"{process_string_fn(pretrained)}_"
    name += f"{process_string_fn(fine_tune)}_"
    name += f"{process_string_fn(seed)}"
    return name


@dataclass
class Config:
    callbacks: Any = MISSING
    logger: Any = MISSING
    model: Any = MISSING
    datamodule: Any = MISSING
    optimizer: Any = MISSING
    trainer: Any = MISSING
    mode: Any = MISSING
    hydra: Any = MISSING

    resume: bool = False
    checkpoint_path: Optional[str] = None
    # pretty print config at the start of the run using Rich library
    print_config: bool = True

    # disable python warnings if they annoy you
    ignore_warnings: bool = True
    logging_level: str = "INFO"
    prefix: str = "exp"
    # evaluate on test set, using best model weights achieved during training
    # lightning chooses best weights based on metric specified in checkpoint
    # callback
    test_after_training: bool = True
    batch_size: int = 1
    # seed for random number generators in learn2learn_hub, numpy and python.random
    seed: int = 0
    # top level argument that sets all the downstream configs to run an
    # experiment on this many iterations

    # path to original working directory
    # hydra hijacks working directory by changing it to the new log directory
    # so it's useful to have this path as a special variable
    # https://hydra.cc/docs/next/tutorials/basic/running_your_app/working_directory
    root_experiment_dir: str = os.environ["EXPERIMENTS_DIR"]
    # path to folder with data
    data_dir: str = os.environ["DATASET_DIR"]
    defaults: List[Any] = field(default_factory=lambda: defaults)
    overrides: List[Any] = field(default_factory=lambda: overrides)
    name: str = generate_name(
        prefix="${prefix}",
        optimizer="${optimizer._target_}",
        dataset_name="${datamodule.dataset_config.challenge_image_source}",
        model_name="${model.model_name_or_path}",
        pretrained="${model.pretrained}",
        fine_tune="${model.fine_tunable}",
        seed=seed,
    )

    current_experiment_dir: str = "${root_experiment_dir}/${name}"
    code_dir: str = "${hydra:runtime.cwd}"


base_callbacks = dict(
    model_checkpoint_eval=model_checkpoint_eval,
    model_checkpoint_train=model_checkpoint_train,
    model_summary=ModelSummaryConfig(),
    progress_bar=RichProgressBar(),
    lr_monitor=LearningRateMonitor(),
)

wandb_callbacks = dict(
    model_checkpoint_eval=model_checkpoint_eval,
    model_checkpoint_train=model_checkpoint_train,
    model_summary=ModelSummaryConfig(),
    progress_bar=RichProgressBar(),
    lr_monitor=LearningRateMonitor,
    code_upload=UploadCodeAsArtifact(),
    log_grads=LogGrads(),
    log_config=LogConfigInformation(),
)
