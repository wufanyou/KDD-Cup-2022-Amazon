from utils.experiment import *
from pytorch_lightning import Trainer
from omegaconf import DictConfig

__all__ = ["get_trainer"]
__key__ = "trainer"


def get_trainer(cfg: DictConfig) -> Trainer:
    logger = get_logger(cfg)
    checkpoint_callback = get_saver(cfg)
    args = dict(cfg[__key__])
    args = {str(k).lower(): v for k, v in args.items()}
    args["logger"] = logger
    args["callbacks"] = [checkpoint_callback]
    return Trainer(**args)
