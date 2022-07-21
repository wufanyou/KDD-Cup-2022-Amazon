from pytorch_lightning import LightningModule
from omegaconf import DictConfig
from utils.model import get_model

from utils.optimizer import get_optimizer
from utils.loss import get_loss
from utils.metric import get_metric

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from typing import Tuple, Union, List
import glob
import re
import numpy as np

__all__ = ["get_lighting"]
__key__ = "lighting"


def get_lighting(cfg: DictConfig, checkpoint: str = "") -> LightningModule:
    module: LightningModule = eval(cfg[__key__].version)
    if checkpoint == "" and cfg.checkpoint != "":
        checkpoint = cfg.checkpoint
    if checkpoint == "":
        model = module(cfg)
    elif checkpoint == "last":
        model_dir = cfg.disk.model_dir
        checkpoint = glob.glob(f"{model_dir}/{cfg.experiment.name}*last.ckpt")[0]
        print(f"load: {checkpoint}")
        model = module.load_from_checkpoint(checkpoint, cfg=cfg)
    elif checkpoint == "best":
        model_dir = cfg.disk.model_dir
        checkpoint = glob.glob(f"{model_dir}/{cfg.experiment.name}*.ckpt")  # type: ignore
        score = [float(re.findall("score=0\.\d+", x)[0][6:]) for x in checkpoint]
        idx = np.argmax(score)
        checkpoint = checkpoint[idx]
        print(f"load: {checkpoint}")
        model = module.load_from_checkpoint(checkpoint, cfg=cfg)
    else:
        print(f"load: {checkpoint}")
        model = module.load_from_checkpoint(checkpoint, cfg=cfg)  # type: ignore
    return model


class BaseLightingModule(LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.model = get_model(cfg)
        self.metrics = get_metric(cfg)
        self.loss_fn = get_loss(cfg)

    def forward(
        self, input_ids, token_type_ids, attention_mask, return_dict=True, **kwargs
    ):
        output = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )
        return output

    def training_step(self, batch: dict, batch_idx: Tensor) -> Tensor:
        y_hat = self(**batch["features"]).logits
        loss = self.loss_fn(y_hat, batch["label"])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: dict, batch_idx: Tensor) -> None:
        y_hat = self(**batch["features"]).logits
        loss = self.metrics(y_hat, batch["label"])
        return loss

    def validation_epoch_end(self, outputs: list) -> None:
        self.log("score", self.metrics.compute())  # type: ignore

    def configure_optimizers(
        self,
    ) -> Union[Tuple[List[Optimizer], Union[List[LambdaLR], List[dict]]], Optimizer]:
        optimizer, scheduler = get_optimizer(self.cfg, self.model)
        if scheduler is not None:
            return [optimizer], [scheduler]
        else:
            return optimizer

class BiEncoderLightingModule(BaseLightingModule):
    def forward(
        self,
        query,
        product,
        use_embeding = False,
        return_dict=True,
         **kwargs,
    ): 
        output = self.model(
            query=query,
            product=product,
            use_embeding=use_embeding,
            return_dict=return_dict,
        )
        return output

class ConcatLightingModule(BaseLightingModule):
    def forward(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        speical_token_pos,
        return_dict=True,
        **kwargs,
    ):
        output = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            speical_token_pos=speical_token_pos,
            return_dict=return_dict,
        )
        return output


class KDConcatLightingModule(ConcatLightingModule):
    def training_step(self, batch: dict, batch_idx: Tensor) -> Tensor:
        y_hat = self(**batch["features"]).logits
        loss = (
            self.loss_fn(y_hat, batch["label"]) * self.cfg.loss.kd_weights[0]
        )  ## pytorch > 1.11
        for i in range(1, len(self.cfg.loss.kd_weights)):
            loss += (
                self.loss_fn(y_hat, batch["kd"][:, i - 1]) * self.cfg.loss.kd_weights[i]
            )
        self.log("train_loss", loss)
        return loss

class KDConcatLightingModuleV2(ConcatLightingModule):
    def training_step(self, batch: dict, batch_idx: Tensor) -> Tensor:
        y_hat = self(**batch["features"]).logits
        loss = (
            self.loss_fn(y_hat, batch["label"]) * self.cfg.loss.kd_weights[0]
        )  ## pytorch > 1.11
        loss += F.mse_loss(y_hat, batch["kd"][:, 0]) * self.cfg.loss.kd_weights[1]
        self.log("train_loss", loss)
        return loss


class KDBiEncoderLightingModuleV2(BiEncoderLightingModule):
    def training_step(self, batch: dict, batch_idx: Tensor) -> Tensor:
        y_hat = self(**batch["features"]).logits
        loss = (
            self.loss_fn(y_hat, batch["label"]) * self.cfg.loss.kd_weights[0]
        )  ## pytorch > 1.11
        loss += F.mse_loss(y_hat, batch["kd"][:, 0]) * self.cfg.loss.kd_weights[1]
        self.log("train_loss", loss)
        return loss

class ConcatExtraLightingModule(BaseLightingModule):
    def forward(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        speical_token_pos,
        extra,
        return_dict=True,
        **kwargs,
    ):
        output = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            speical_token_pos=speical_token_pos,
            extra=extra,
            return_dict=return_dict,
        )
        return output


class WeightedConcatLightingModule(ConcatLightingModule):
    def training_step(self, batch: dict, batch_idx: Tensor) -> Tensor:
        y_hat = self(**batch["features"]).logits
        loss = self.loss_fn(y_hat, batch["label"])
        weight = batch["meta"]["sample_length"]
        weight = 1 / weight
        weight = weight / weight.sum()
        loss = (loss * weight).mean()
        self.log("train_loss", loss)
        return loss


# adv training modify from # https://github.com/antmachineintelligence/Feedback_1st
class AdvLightingModule(BaseLightingModule):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.adv_args = cfg[__key__].adv_args
        self.adv_backup = {}
        self.adv_backup_eps = {}
        self.automatic_optimization = False

    def adv_save(self):
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_args.adv_param in name
            ):
                if name not in self.adv_backup:
                    self.adv_backup[name] = param.data.clone()
                    grad_eps = self.adv_args.adv_eps * param.abs().detach()
                    self.adv_backup_eps[name] = (
                        self.adv_backup[name] - grad_eps,
                        self.adv_backup[name] + grad_eps,
                    )

    def adv_attack(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_args.adv_param in name
            ):
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_args.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.adv_backup_eps[name][0]), self.adv_backup_eps[name][1]  # type: ignore
                    )

    def adv_restore(
        self,
    ):
        for name, param in self.model.named_parameters():
            if name in self.adv_backup:
                param.data = self.adv_backup[name]
        self.adv_backup = {}
        self.adv_backup_eps = {}

    def configure_optimizers(
        self,
    ) -> Union[Tuple[List[Optimizer], Union[List[LambdaLR], List[dict]]], Optimizer]:
        optimizer, scheduler = get_optimizer(self.cfg, self.model)
        if scheduler is not None:
            return [optimizer], [scheduler]
        else:
            return optimizer

    def training_step(self, batch: dict, batch_idx: Tensor) -> Tensor:
        opt = self.optimizers()  # type: ignore
        scheduler = self.lr_schedulers()
        opt.zero_grad()  # type: ignore
        y_hat = self(**batch["features"]).logits
        loss = self.loss_fn(y_hat, batch["label"])
        
        self.manual_backward(loss)

        # perform one step adv training
        if self.global_step >= self.adv_args.start_steps:
            self.adv_save()
            self.adv_attack()
            y_hat = self(**batch["features"]).logits
            loss = self.loss_fn(y_hat, batch["label"])
            self.manual_backward(loss)
            self.adv_restore()

        torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.adv_args.adv_grad_clip)  # type: ignore

        opt.step()  # type: ignore
        scheduler.step()  # type: ignore
        self.log("train_loss", loss)

        return loss


class CatAdvLightingModule(AdvLightingModule):
    def forward(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        speical_token_pos,
        return_dict=True,
        **kwargs,
    ):
        output = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            speical_token_pos=speical_token_pos,
            return_dict=return_dict,
        )
        return output


class KdCatAdvLightingModule(CatAdvLightingModule):
    def training_step(self, batch: dict, batch_idx: Tensor) -> Tensor:
        opt = self.optimizers()  # type: ignore
        scheduler = self.lr_schedulers()
        opt.zero_grad()  # type: ignore
        y_hat = self(**batch["features"]).logits
        loss = (
            self.loss_fn(y_hat, batch["label"]) * self.cfg.loss.kd_weights[0]
        )
        loss += F.mse_loss(y_hat, batch["kd"][:, 0]) * self.cfg.loss.kd_weights[1]
        self.manual_backward(loss)

        # perform one step adv training
        if self.global_step >= self.adv_args.start_steps:
            self.adv_save()
            self.adv_attack()
            y_hat = self(**batch["features"]).logits
            loss = self.loss_fn(y_hat, batch["label"])
            self.manual_backward(loss)
            self.adv_restore()
        torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.adv_args.adv_grad_clip)  # type: ignore
        opt.step()  # type: ignore
        scheduler.step()  # type: ignore
        self.log("train_loss", loss)
        return loss