from transformers.models.auto.tokenization_auto import AutoTokenizer
from omegaconf import DictConfig, ListConfig
from typing import Union
from .cocolm import COCOLMTokenizer

__all__ = ["get_tokenizer"]
__key__ = 'tokenizer'


def get_tokenizer(cfg: Union[DictConfig, ListConfig]):
    if "cocolm" in cfg.model.name:
        return COCOLMTokenizer.from_pretrained(cfg.model.name, use_fast=cfg[__key__].use_fast)
    else:
        return AutoTokenizer.from_pretrained(cfg.model.name, use_fast=cfg[__key__].use_fast)
