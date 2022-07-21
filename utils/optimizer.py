# Created by fw at 12/31/20
import math
import torch.nn as nn
import torch.optim as optim
import transformers
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from typing import Tuple, Optional


__all__ = ["get_optimizer"]
__key__ = "optimizer"


def get_optimizer(
    cfg: DictConfig, model: nn.Module
) -> Tuple[Optimizer, Optional[LambdaLR]]:
    args = dict(cfg[__key__].args)
    args = {str(k).lower(): v for k, v in args.items()}
    optimizer = eval(f"{cfg[__key__].version}")

    param_optimizer = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": cfg[__key__].weight_decay,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = optimizer(optimizer_grouped_parameters, **args)

    if cfg[__key__].scheduler.use:
        scheduler = eval(cfg[__key__].scheduler.version)
        args = dict(cfg[__key__].scheduler.args)
        args = {str(k).lower(): v for k, v in args.items()}
        args["optimizer"] = optimizer
        if "num_training_steps" not in args:
            args["num_training_steps"] = cfg["trainer"].max_steps
        scheduler = scheduler(**args)
        scheduler = {"scheduler": scheduler, "interval": "step"}
    else:
        scheduler = None
    return optimizer, scheduler  # type: ignore


# Modify from https://huggingface.co/transformers/
def linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# Modify from https://huggingface.co/transformers/
def cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    cosine_schedule_with_warmup
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# Modify from https://huggingface.co/transformers/
def exponent_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    exponent: float = 1 - 2e-3,
    step: int = 10,
    last_epoch: int = -1,
) -> LambdaLR:
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            return exponent ** ((current_step - num_warmup_steps) // step)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def linear_schedule_with_warmup_and_station(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_station_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step < num_station_steps:
            return 1.0
        else:
            return max(
                0.0,
                float(num_training_steps - current_step)
                / float(max(1, num_training_steps - num_station_steps)),
            )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def cosine_with_hard_restarts_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int = 1,
    last_epoch: int = -1,
    min_learning_rate:float = 0.0
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        if progress >= 1.0:
            return min_learning_rate
        return max(
            min_learning_rate,
            0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)
