import os

import dotenv
import hydra
import rich

from capit.base.utils import pretty_config
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from rich.traceback import install


# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir

dotenv.load_dotenv(override=True)
install(show_locals=False, extra_lines=1, word_wrap=True, width=350)


def collect_config_store():
    from capit.configs.config_tree import (
        wandb_callbacks,
        base_callbacks,
        Config,
    )
    from capit.configs.datamodules import InstagramImageTextMultiModalDataModuleConfig
    from capit.configs.hydra import add_hydra_configs
    from capit.configs.loggers import WeightsAndBiasesLoggerConfig
    from capit.configs.mode import BaseMode
    from capit.configs.models import CLIPImageTextMultiModalDatasetConfig
    from capit.configs.optimizers import AdamWOptimizerConfig
    from capit.configs.trainers import BaseTrainer, DPTrainer, DDPTrainer, MPSTrainer

    config_store = ConfigStore.instance()
    config_store.store(name="config", node=Config)

    config_store.store(
        group="callbacks",
        name="base",
        node=base_callbacks,
    )

    config_store.store(
        group="callbacks",
        name="wandb",
        node=wandb_callbacks,
    )

    config_store.store(
        group="logger",
        name="wandb",
        node=dict(wandb_logger=WeightsAndBiasesLoggerConfig()),
    )

    config_store.store(
        group="model",
        name="clip-image-text",
        node=CLIPImageTextMultiModalDatasetConfig,
    )

    config_store.store(
        group="datamodule",
        name="InstagramImageTextMultiModal",
        node=InstagramImageTextMultiModalDataModuleConfig,
    )

    config_store.store(group="trainer", name="base", node=BaseTrainer)
    config_store.store(group="trainer", name="gpu-dp", node=DPTrainer)
    config_store.store(group="trainer", name="gpu-ddp", node=DDPTrainer)
    config_store.store(group="trainer", name="mps", node=MPSTrainer)

    config_store = add_hydra_configs(config_store)

    config_store.store(
        group="mode",
        name="base",
        node=BaseMode(),
    )

    config_store.store(group="optimizer", name="AdamW", node=AdamWOptimizerConfig)

    return config_store


config_store = collect_config_store()


@hydra.main(version_base=None, config_name="config")
def main(config: DictConfig):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from capit.base import utils
    from capit.train_eval import train_eval

    # A couple of optional utilities:
    # - disabling python warnings
    # - forcing debug-friendly configuration
    # - verifying experiment name is set when running in experiment mode
    # You can safely get rid of this line if you don't want those
    utils.extras(config)
    os.environ["WANDB_PROGRAM"] = config.code_dir
    # Pretty print config using Rich library
    if config.get("print_config"):
        config_tree = pretty_config(config, resolve=True)
        rich.print(config_tree)

        with open("config_tree.log", "w") as fp:
            rich.print(config_tree, file=fp)

    return train_eval(config)


if __name__ == "__main__":
    main()
