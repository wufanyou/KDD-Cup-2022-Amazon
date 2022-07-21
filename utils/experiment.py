from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import yaml

__all__ = ["get_logger", "get_saver"]
__key__ = "experiment"


def get_logger(cfg: DictConfig) -> LightningLoggerBase:
    if cfg[__key__].logger.version == "MLFlowLogger":
        from pytorch_lightning.loggers import MLFlowLogger

        tags = {"version": cfg[__key__].name}
        logger = MLFlowLogger(
            experiment_name=cfg[__key__].logger.experiment_name,
            tracking_uri=cfg[__key__].logger.tracking_uri,
            tags=tags,
        )
    elif cfg[__key__].logger.version == "WandbLogger":
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(
            name=cfg[__key__].name,
            project=cfg[__key__].logger.experiment_name,
            config=yaml.load(OmegaConf.to_yaml(cfg), Loader=yaml.FullLoader),
        )
    else:
        assert False, "only support MLFlow or Wandb"
    return logger


def get_saver(cfg: DictConfig) -> ModelCheckpoint:
    args = dict(cfg[__key__].saver)
    args["filename"] = args["filename"].format(experiment=cfg[__key__].name)
    args = {str(k).lower(): v for k, v in args.items()}
    args["dirpath"] = cfg.disk.model_dir
    saver = ModelCheckpoint(**args)
    if cfg.train_all:
        saver.CHECKPOINT_NAME_LAST = cfg.experiment.name +'-last'
    else:
        saver.CHECKPOINT_NAME_LAST = args["filename"] + "-last"  # type: ignore
    return saver


