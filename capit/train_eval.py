import os
import pathlib
from typing import List, Optional

import hydra
import pytorch_lightning
import torch
import wandb
from hydra_zen import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback, LightningDataModule, Trainer, seed_everything
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.tuner.tuning import Tuner
from wandb.util import generate_id
from wandbless.checkpointing import StatelessCheckpointingWandb

from capit.base import utils
from capit.base.utils.typing_utils import get_module_import_path
from capit.configs.callbacks import LogConfigInformation
from capit.lightning.train_eval_agent import TrainingEvaluationAgent

log = utils.get_logger(__name__)


def checkpoint_setup(config, restore_agent: StatelessCheckpointingWandb):
    checkpoint_path = None
    experiment_dir = pathlib.Path(f"{config.current_experiment_dir}")

    if config.resume:

        if not experiment_dir.exists():
            experiment_dir.mkdir(exist_ok=True, parents=True)

        log.info("Downloading latest checkpoint from wandb")
        try:
            checkpoint_path = restore_agent.restore_latest(
                store_dir=os.environ["MODEL_DIR"], model_name=config.name
            )
            log.info("Continue from existing checkpoint")
        except Exception:
            log.exception(
                "Failed to download latest checkpoint from wandb, starting from scratch"
            )
            checkpoint_path = None

        log.info(checkpoint_path)

    else:

        log.info("Starting from scratch")
        if not experiment_dir.exists():
            experiment_dir.mkdir(exist_ok=True, parents=True)

    return checkpoint_path


def train_eval(config: DictConfig):
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    wandb_checkpointer = StatelessCheckpointingWandb(
        job_type=config.logger.wandb.job_type,
        dir=config.logger.wandb.save_dir,
        config=config,
        project=config.logger.wandb.project,
        entity=os.environ["WANDB_ENTITY"],
        id=config.logger.wandb.id,
    )

    if config.get("seed"):
        seed_everything(config.seed, workers=True)
    # --------------------------------------------------------------------------------
    # Create or recover checkpoint path to resume from
    checkpoint_path = checkpoint_setup(config, restore_agent=wandb_checkpointer)
    # --------------------------------------------------------------------------------
    # Instantiate Lightning DataModule for task
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    # log information regarding data module to be instantiated -- particularly the class name that is stored in _target_
    datamodule: LightningDataModule = instantiate(config.datamodule, _recursive_=False)
    # List in comments all possible datamodules/datamodule configs
    datamodule.setup(stage="fit")
    # datamodule_pretty_dict_tree = generate_config_tree(
    #     config=datamodule.__dict__, resolve=True
    # )
    log.info(f"Datamodule <{config.datamodule._target_}> instantiated")
    # --------------------------------------------------------------------------------
    # Instantiate Lightning TrainingEvaluationAgent for task
    log.info(f"Instantiating model <{config.model._target_}>")

    train_eval_agent: TrainingEvaluationAgent = TrainingEvaluationAgent(
        model_config=config.model,
        optimizer_config=config.optimizer,
        datamodule=datamodule,
    )

    # --------------------------------------------------------------------------------
    # Instantiate Lightning Callbacks
    # --------------------------------------------------------------------------------
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                if "LogConfigInformation" in cb_conf["_target_"]:
                    log.info(
                        f"Instantiating config collection callback <{cb_conf._target_}>"
                    )
                    cb_conf["config"] = OmegaConf.to_container(config, resolve=True)
                    callbacks.append(
                        instantiate(
                            cb_conf,
                            _recursive_=False,
                        )
                    )

                elif "SaveCheckpointsWandb" in cb_conf["_target_"]:
                    log.info(f"Instantiating <{cb_conf._target_}>")
                    cb_conf["wandb_checkpointer"] = wandb_checkpointer
                    callbacks.append(
                        instantiate(
                            cb_conf,
                            _recursive_=False,
                        )
                    )
                else:
                    log.info(f"Instantiating callback <{cb_conf._target_}>")
                    callbacks.append(instantiate(cb_conf))

    # --------------------------------------------------------------------------------
    # Instantiate Experiment Logger
    # --------------------------------------------------------------------------------
    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(instantiate(lg_conf))

    # --------------------------------------------------------------------------------
    # Instantiate Lightning Trainer
    # --------------------------------------------------------------------------------
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = instantiate(
        config.trainer,
        callbacks=callbacks,
        logger=logger,
        _convert_="partial",
    )

    # --------------------------------------------------------------------------------
    # If auto_scale_batch_size is set, we need to tune the batch size using
    # the Lightning Tuner class, starting from given batch size and increasing
    # in powers of 2
    if config.trainer.auto_scale_batch_size:
        tuner = Tuner(trainer)
        new_batch_size = tuner.scale_batch_size(
            train_eval_agent,
            datamodule=datamodule,
            mode="power",
            init_val=2 * torch.cuda.device_count(),
        )
        datamodule.batch_size = new_batch_size
        config.datamodule.batch_size = new_batch_size

    # --------------------------------------------------------------------------------
    # Start training
    if config.mode.fit:
        log.info("Starting training!")
        trainer.validate(
            model=train_eval_agent,
            datamodule=datamodule,
            ckpt_path=checkpoint_path,
        )

        trainer.fit(
            model=train_eval_agent,
            datamodule=datamodule,
            ckpt_path=checkpoint_path,
        )

    # --------------------------------------------------------------------------------
    # Start evaluation on test set
    if config.mode.test and not config.trainer.get("fast_dev_run"):
        datamodule.setup(stage="test")

        log.info("Starting testing ! 🧪")

        if config.mode.fit is False:
            test_results = trainer.test(
                model=train_eval_agent,
                datamodule=datamodule,
                verbose=False,
                ckpt_path=checkpoint_path,
            )
        else:
            test_results = trainer.test(
                model=train_eval_agent,
                datamodule=datamodule,
                verbose=False,
            )

        log.info(
            f"Testing results can be found in the wandb log: {wandb.run.url}, "
            f"please only check that when finalizing your conclusions, "
            f"otherwise you are at risk of subconsciosuly biasing your "
            f"results 🚨"
        )
        for logger_instance in logger:
            if isinstance(logger_instance, pytorch_lightning.loggers.wandb.WandbLogger):
                wandb.log(test_results[0], step=0)
    # Make sure everything closed properly
    log.info("Finalizing! 😺")
    # Print path to best checkpoint
    if not config.trainer.get("fast_dev_run"):
        log.info(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")

    wandb.finish(quiet=False)
