from typing import List, Dict

from pytorch_lightning import Callback, Trainer, LightningModule


# class RunValidationOnTrainStart(Callback):
#     def __init__(self):
#         super().__init__()
#
#     def on_train_epoch_start(
#         self, trainer: Trainer, pl_module: LightningModule
#     ) -> None:
#         trainer.validate(ckpt_path="None", datamodule=trainer._data_connector.)
