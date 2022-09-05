import logging

from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities.model_summary import ModelSummary


class PostBuildSummary(Callback):
    """
    Callback to log model summary after model is built
    """

    def __init__(self, max_depth):
        super(PostBuildSummary, self).__init__()
        self.max_depth = max_depth

    def on_sanity_check_end(
        self, trainer: "Trainer", pl_module: "LightningModule"
    ) -> None:
        summary = ModelSummary(model=pl_module, max_depth=self.max_depth)
        logging.info(summary)
