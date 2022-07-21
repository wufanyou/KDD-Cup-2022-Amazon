from omegaconf import DictConfig
import torch.nn as nn
from torchmetrics.classification.f_beta import F1Score

__all__ = ["get_metric"]
__key__ = "metric"


def get_metric(cfg: DictConfig) -> nn.Module:
    args = dict(cfg[__key__].args)
    args = {str(k).lower(): v for k, v in args.items()}
    metric_fn = eval(cfg[__key__].version)(**args)
    return metric_fn
