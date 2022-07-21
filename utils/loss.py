import torch.nn as nn
from omegaconf import DictConfig

__all__ = ["get_loss"]
__key__ = "loss"


def get_loss(cfg: DictConfig) -> nn.Module:
    args = dict(cfg[__key__].args)
    args = {str(k).lower(): v for k, v in args.items()}
    loss_fn = eval(cfg[__key__].version)(**args)
    return loss_fn

