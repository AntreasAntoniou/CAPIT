from typing import Any, Dict, List

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only

from tali.utils.storage import (
    google_storage_rsync_gs_to_local,
    google_storage_rsync_local_to_gs,
)


class GoogleStorageBucketRSyncClient(Callback):
    def __init__(
        self,
        bucket_name: str = None,
        experiments_root_dir: str = None,
        experiment_name: str = None,
        exclude_list: List[str] = False,
        options_list: List[str] = None,
        resume: bool = True,
    ):
        super().__init__()
        self.bucket_name = bucket_name
        self.experiments_root_dir = experiments_root_dir
        self.experiment_name = experiment_name
        self.exclude_list = exclude_list
        self.options_list = options_list
        self.resume = resume

    @rank_zero_only
    def on_save_checkpoint(
        self,
        trainer: "pl.CustomTrainer",
        pl_module: "pl.LightningModule",
        checkpoint: Dict[str, Any],
    ) -> Dict[str, Any]:
        google_storage_rsync_local_to_gs(
            bucket_name=self.bucket_name,
            experiments_root_dir=self.experiments_root_dir,
            experiment_name=self.experiment_name,
            exclude_list=self.exclude_list,
            options_list=self.options_list,
        )

    @rank_zero_only
    def on_load_checkpoint(
        self,
        trainer: "pl.CustomTrainer",
        pl_module: "pl.LightningModule",
        callback_state: Dict[str, Any],
    ) -> None:
        if self.resume:
            google_storage_rsync_gs_to_local(
                bucket_name=self.bucket_name,
                experiments_root_dir=self.experiments_root_dir,
                experiment_name=self.experiment_name,
                exclude_list=self.exclude_list,
                options_list=self.options_list,
            )
