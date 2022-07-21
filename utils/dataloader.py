from typing import Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, ConcatDataset
from omegaconf import DictConfig
from utils.dataset import get_dataset, BaseDataset
from .prepare import prepare_data
import os
from utils.prepare import get_prefix
from utils.config import get_cfg

__ALL__ = ["get_dataloader"]
__key__ = "dataloader"


def get_dataloader(cfg: DictConfig) -> LightningDataModule:
    dataloader = eval(cfg[__key__].version)(cfg=cfg)
    return dataloader


class BaseDataLoader(LightningDataModule):
    r"""BaseDataLoader
    Args:
        cfg (OmegaConf): global config file
    """

    def __init__(self, cfg: DictConfig) -> None:
        super(BaseDataLoader, self).__init__()
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self) -> None:

        if self.cfg.locale_all == False:
            prepare_data(self.cfg)
            local_data_path = self.cfg.disk.local_dir
            host_data_path = self.cfg.disk.output_dir
            prefix = get_prefix(self.cfg)
            host_data = f"{host_data_path}/{prefix}.h5"
            if local_data_path!=host_data_path:
                os.makedirs(local_data_path, exist_ok=True)
                os.system(f"cp -rf {host_data} {local_data_path}/")
        else:
            local_data_path = self.cfg.disk.local_dir
            for c in self.cfg.other_cfg:
                c = get_cfg(c)
                host_data_path = c.disk.output_dir
                prefix = get_prefix(c)
                host_data = f"{host_data_path}/{prefix}.h5"
                if local_data_path!=host_data_path:
                    os.makedirs(local_data_path, exist_ok=True)
                    os.system(f"cp -rf {host_data} {local_data_path}/")

    def setup(self, stage=None) -> None:

        if stage == "fit" or stage is None:
            self.val_dataset = get_dataset(self.cfg, "val")
            self.train_dataset = get_dataset(self.cfg, "train")

        if stage == "test" or stage is None:
            self.test_dataset = get_dataset(self.cfg, "test")
            self.val_dataset = get_dataset(self.cfg, "val")     

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader("train")

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val")

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test")

    def get_dataloader(self, split: str) -> DataLoader:
        assert split in ["train", "test", "val"]
        batch_size = self.cfg[__key__].batch_size[split]
        num_workers = self.cfg[__key__].num_workers
        dataset = eval(f"self.{split}_dataset") 
        

        if self.cfg[__key__].override_drop_last:
            drop_last = False
        else:
            drop_last = split == "train"

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=True,
            worker_init_fn=dataset.worker_init_fn,
            collate_fn=dataset.collate_fn,
            persistent_workers=True,
        )
        return loader

