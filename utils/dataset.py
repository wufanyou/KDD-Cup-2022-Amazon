import atexit
import os
from typing import Dict, List, Optional, Tuple
import pandas as pd

import h5py
import numpy as np
import torch
from torch.utils.data import ConcatDataset
from numpy import ndarray, product
from omegaconf import DictConfig
from sklearn.model_selection import KFold
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, get_worker_info
from .tokenizer import get_tokenizer
from .prepare import __map__, get_prefix
import bisect

__key__ = "dataset"
__all__ = ["get_dataset"]


def get_dataset(cfg: DictConfig, split: str) -> Dataset:
    if cfg.locale_all:
        dataset = AllLocaleDataset(cfg, split)
    else:
        dataset = eval(cfg[__key__].version)
        dataset = dataset(cfg, split)
    return dataset


class BaseDataset(Dataset):
    def __init__(self, cfg: DictConfig, split: str) -> None:
        assert split in ["train", "test", "val"]
        self.cfg = cfg
        self.split = split
        self.split_dataset = ["train", "test"][self.split == "test"]
        if self.split_dataset == "test" and cfg[__key__].other:
            self.split_dataset = "other"
        self.max_length = cfg.model.max_length
        prefix = get_prefix(self.cfg)
        if os.path.exists(f"{cfg.disk.local_dir}/{prefix}.h5"):
            filename = f"{cfg.disk.local_dir}/{prefix}.h5"
        elif os.path.exists(f"{cfg.disk.output_dir}/{prefix}.h5"):
            filename = f"{cfg.disk.output_dir}/{prefix}.h5"
        else:
            raise FileNotFoundError

        self.filename = filename
        self._split()
        self.used_col = cfg[__key__].used_col
        self._chr_num_map()

    def _chr_num_map(self) -> None:
        t = get_tokenizer(self.cfg)
        m = {}
        for i in range(10):
            token = t(str(i))["input_ids"]  # type: ignore
            m[str(i)] = token[1]  # type: ignore

        for i in range(97, 97 + 26):
            token = t(chr(i))["input_ids"]  # type: ignore
            m[chr(i)] = token[1]  # type: ignore

        for i in range(65, 65 + 26):
            token = t(chr(i))["input_ids"]  # type: ignore
            m[chr(i)] = token[1]  # type: ignore

        m["cls"] = token[0]  # type: ignore
        m["sep"] = token[-1]  # type: ignore
        self.token_map = m

    def _split(self) -> None:
        with h5py.File(self.filename, "r", libver="latest", swmr=True) as ds:
            keys = list(ds[self.split_dataset].keys())  # type: ignore
            if self.cfg.sample_task_1:
                f = set(pd.read_csv(self.cfg.disk.sample_file)['query_id'])
                keys = [x for x in keys if x in f]

            if self.cfg.train_all:
                if not self.cfg.val_all:
                    if self.split == "val":
                        keys = keys[:128]
            else:
                if self.split not in ("test", "other"):
                    cv = KFold(
                        n_splits=self.cfg.total_fold,
                        random_state=self.cfg.seed,
                        shuffle=True,
                    )
                    idx = list(cv.split(keys))[self.cfg.fold][self.split == "val"]
                    keys = [keys[x] for x in idx]

            samples = []
            sample_length = {}
            for k in keys:
                length = len(ds[self.split_dataset][k]["product_id"])  # type: ignore
                samples += [(k, x) for x in range(length)]
                sample_length[k] = length
            self.sample_length = sample_length
            self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def cleanup(self) -> None:
        self.database.close()

    def worker_init(self) -> None:
        self.database = h5py.File(self.filename, "r", libver="latest", swmr=True)
        atexit.register(self.cleanup)

    @staticmethod
    def worker_init_fn(worker_id):
        worker_info = get_worker_info()
        dataset = worker_info.dataset  # type: ignore
        dataset.worker_init()

    @staticmethod
    def collate_fn(batch):
        ...


class Task1Dataset(BaseDataset):
    @staticmethod
    def collate_fn(batch):
        pass


def _process_encoding(
    arr: ndarray, encode_map: dict, name="query", token_map: Optional[dict] = None
) -> Tensor:
    arr = np.array(arr)
    if name == "query":
        arr = np.insert(arr, 1, encode_map[name])
    elif name == "product_id":
        arr = str(arr)[2:-1]  # type: ignore
        arr = [token_map[x] for x in arr]  # type: ignore
        arr = [encode_map[name]] + arr + [token_map["sep"]]  # type: ignore
    elif name == "index":
        arr = str(arr[0])  # type: ignore
        arr = [token_map[x] for x in arr]  # type: ignore
        arr = [encode_map[name]] + arr + [token_map["sep"]]  # type: ignore
    else:
        arr[0] = encode_map[name]
    tensor = torch.tensor(arr, dtype=torch.long)
    return tensor


class Task2Dataset(BaseDataset):
    def __getitem__(self, index) -> Tuple:
        query_id, idx = self.samples[index]
        product_id = self.database[self.split_dataset][query_id]["product_id"][idx]  # type: ignore
        example_id = self.database[self.split_dataset][query_id]["example_id"][idx]  # type: ignore
        dataset = torch.tensor([self.database[self.split_dataset][query_id]["dataset"][idx]], dtype=torch.long)  # type: ignore
        esci_label = torch.tensor([self.database[self.split_dataset][query_id]["esci_label"][idx]], dtype=torch.long)  # type: ignore
        query_encode = _process_encoding(self.database[self.split_dataset][query_id]["query"], encode_map=self.cfg.model.encode)  # type: ignore
        input_ids = [query_encode]
        for name in self.used_col:
            if name == "product_id":
                input_ids.append(_process_encoding(product_id, self.cfg.model.encode, name, self.token_map))  # type: ignore
            else:
                arr = self.database["product_catalogue"][product_id][name]  # type: ignore
                input_ids.append(_process_encoding(arr, self.cfg.model.encode, name, self.token_map))  # type: ignore
        input_ids = torch.cat(input_ids)  # type: ignore

        if len(input_ids) > self.max_length:
            tail = input_ids[-1]
            input_ids = input_ids[: self.max_length]
            input_ids[-1] = tail

        token_type_ids = torch.zeros_like(input_ids)
        attention_mask = torch.ones_like(input_ids)

        feature = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "extra": dataset,
        }

        meta = {
            "product_id": product_id,
            "query_id": query_id,
            "example_id": example_id,
            "pad_token_id": self.cfg.model.pad_token_id,
            "sample_length": self.sample_length[query_id],
        }

        return feature, esci_label, meta

    @staticmethod
    def collate_fn(batch: List) -> dict:
        features = {}
        pad_token_id = batch[0][2]["pad_token_id"]
        features["input_ids"] = pad_sequence(
            [x[0]["input_ids"] for x in batch],
            batch_first=True,
            padding_value=pad_token_id,
        )
        features["token_type_ids"] = pad_sequence(
            [x[0]["token_type_ids"] for x in batch],
            batch_first=True,
        )
        features["attention_mask"] = pad_sequence(
            [x[0]["attention_mask"] for x in batch],
            batch_first=True,
        )
        features["extra"] = torch.cat([x[0]["extra"] for x in batch])
        label = torch.cat([x[1] for x in batch])
        meta = {}
        meta["product_id"] = [x[2]["product_id"] for x in batch]
        meta["example_id"] = [x[2]["example_id"] for x in batch]
        meta["query_id"] = [x[2]["query_id"] for x in batch]
        meta["sample_length"] = torch.tensor(
            [x[2]["sample_length"] for x in batch], dtype=torch.float
        )
        return {"features": features, "label": label, "meta": meta}


class Task2DatasetConCat(BaseDataset):
    def __getitem__(self, index) -> Tuple:
        query_id, idx = self.samples[index]
        product_id = self.database[self.split_dataset][query_id]["product_id"][idx]  # type: ignore
        example_id = self.database[self.split_dataset][query_id]["example_id"][idx]  # type: ignore
        dataset = torch.tensor([self.database[self.split_dataset][query_id]["dataset"][idx]], dtype=torch.long)[None]  # type: ignore
        esci_label = torch.tensor([self.database[self.split_dataset][query_id]["esci_label"][idx]], dtype=torch.long)  # type: ignore
        query_encode = _process_encoding(self.database[self.split_dataset][query_id]["query"], encode_map=self.cfg.model.encode)  # type: ignore
        input_ids = [query_encode]
        input_ids_pos = [1]
        for name in self.used_col:
            if name == "product_id":
                input_ids.append(_process_encoding(product_id, self.cfg.model.encode, name, self.token_map))  # type: ignore
            else:
                arr = self.database["product_catalogue"][product_id][name]  # type: ignore
                input_ids.append(_process_encoding(arr, self.cfg.model.encode, name, self.token_map))  # type: ignore
            input_ids_pos.append(sum(len(x) for x in input_ids[:-1]))
        input_ids = torch.cat(input_ids)  # type: ignore
        for i in range(len(input_ids_pos)):
            if input_ids_pos[i] >= self.max_length:
                if input_ids_pos[-2] < self.max_length:
                    input_ids_pos[i] = input_ids_pos[-2]
                elif input_ids_pos[1] < self.max_length:
                    input_ids_pos[i] = input_ids_pos[1]
                else:
                    input_ids_pos[i] = self.max_length - 1

        input_ids_pos = torch.tensor(input_ids_pos, dtype=torch.long)[None]

        if len(input_ids) > self.max_length:
            tail = input_ids[-1]
            input_ids = input_ids[: self.max_length]
            input_ids[-1] = tail

        token_type_ids = torch.zeros_like(input_ids)
        attention_mask = torch.ones_like(input_ids)

        feature = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "speical_token_pos": input_ids_pos,
            "extra": dataset,
        }

        meta = {
            "product_id": product_id,
            "query_id": query_id,
            "example_id": example_id,
            "pad_token_id": self.cfg.model.pad_token_id,
            "sample_length": self.sample_length[query_id],
        }

        return feature, esci_label, meta

    @staticmethod
    def collate_fn(batch: List) -> dict:
        features = {}
        pad_token_id = batch[0][2]["pad_token_id"]
        features["input_ids"] = pad_sequence(
            [x[0]["input_ids"] for x in batch],
            batch_first=True,
            padding_value=pad_token_id,
        )
        features["token_type_ids"] = pad_sequence(
            [x[0]["token_type_ids"] for x in batch],
            batch_first=True,
        )
        features["attention_mask"] = pad_sequence(
            [x[0]["attention_mask"] for x in batch],
            batch_first=True,
        )
        features["speical_token_pos"] = torch.cat(
            [x[0]["speical_token_pos"] for x in batch]
        )
        features["extra"] = torch.cat([x[0]["extra"] for x in batch])
        label = torch.cat([x[1] for x in batch])
        meta = {}
        meta["product_id"] = [x[2]["product_id"] for x in batch]
        meta["example_id"] = [x[2]["example_id"] for x in batch]
        meta["query_id"] = [x[2]["query_id"] for x in batch]
        meta["sample_length"] = torch.tensor(
            [x[2]["sample_length"] for x in batch], dtype=torch.float
        )

        output = {"features": features, "label": label, "meta": meta}
        if len(batch[0]) == 4:
            output["kd"] = torch.stack([x[3] for x in batch])
        return output


class KDTask2DatasetConCat(Task2DatasetConCat):
    def __getitem__(self, index) -> Tuple:
        feature, esci_label, meta = super().__getitem__(index)
        if self.split != "train":
            return feature, esci_label, meta
        else:
            example_id = meta["example_id"]
            kd = []
            for f in self.kd:
                kd.append(
                    torch.tensor(
                        f.loc[example_id][["p0", "p1", "p2", "p3"]].values,
                        dtype=torch.float32,
                    )
                )
            kd = torch.stack(kd)
            return feature, esci_label, meta, kd

    def worker_init(self) -> None:
        self.database = h5py.File(self.filename, "r", libver="latest", swmr=True)
        self.kd = [pd.read_csv(f).set_index("example_id") for f in self.cfg.disk.kd]

        atexit.register(self.cleanup)


class AllLocaleDataset(Task2DatasetConCat):
    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, cfg: DictConfig, split: str) -> None:
        datasets = []
        from .config import get_cfg
        for other_cfg in cfg.other_cfg:
            other_cfg = get_cfg(other_cfg)
            other_cfg['total_fold'] = cfg['total_fold']
            other_cfg['fold'] = cfg['fold']
            if split == 'train':
                datasets.append(get_dataset(other_cfg, split))
            else:
                if other_cfg['locale'] != 'us':
                    datasets.append(get_dataset(other_cfg, split))
        self.datasets = datasets
        self.cumulative_sizes = self.cumsum(self.datasets)
        self.cfg = cfg

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    def __len__(self):
        return self.cumulative_sizes[-1]

    def worker_init(self) -> None:
        for d in self.datasets:
            d.worker_init()

    @staticmethod
    def worker_init_fn(worker_id):
        worker_info = get_worker_info()
        dataset = worker_info.dataset  # type: ignore
        dataset.worker_init()


class Task3DatasetConCat(Task2DatasetConCat):
    def __getitem__(self, index) -> Tuple:
        feature, esci_label, meta = super().__getitem__(index)
        esci_label = (esci_label == 2).type(torch.long)
        return feature, esci_label, meta


def _process_encoding_bi(
    arr: ndarray,
    encode_map: dict,
    name="query",
    token_map: Optional[dict] = None,
    is_first=False,
) -> Tensor:
    arr = np.array(arr)
    if name == "query" or is_first:
        arr = np.insert(arr, 1, encode_map[name])
    elif name == "product_id":
        arr = str(arr)[2:-1]  # type: ignore
        arr = [token_map[x] for x in arr]  # type: ignore
        arr = [encode_map[name]] + arr + [token_map["sep"]]  # type: ignore
    elif name == "index":
        arr = str(arr[0])  # type: ignore
        arr = [token_map[x] for x in arr]  # type: ignore
        arr = [encode_map[name]] + arr + [token_map["sep"]]  # type: ignore
    else:
        arr[0] = encode_map[name]
    tensor = torch.tensor(arr, dtype=torch.long)
    return tensor


class BiEncoderDatasetTask2(BaseDataset):
    def __getitem__(self, index) -> Tuple:
        query_id, idx = self.samples[index]
        product_id = self.database[self.split_dataset][query_id]["product_id"][idx]  # type: ignore
        example_id = self.database[self.split_dataset][query_id]["example_id"][idx]  # type: ignore
        dataset = torch.tensor([self.database[self.split_dataset][query_id]["dataset"][idx]], dtype=torch.long)[None]  # type: ignore
        esci_label = torch.tensor([self.database[self.split_dataset][query_id]["esci_label"][idx]], dtype=torch.long)  # type: ignore
        query = _process_encoding_bi(self.database[self.split_dataset][query_id]["query"], encode_map=self.cfg.model.encode)  # type: ignore
        input_ids = []
        for i, name in enumerate(self.used_col):
            if name == "product_id":
                input_ids.append(_process_encoding_bi(product_id, self.cfg.model.encode, name, self.token_map, is_first=i == 0))  # type: ignore
            else:
                arr = self.database["product_catalogue"][product_id][name]  # type: ignore
                input_ids.append(_process_encoding_bi(arr, self.cfg.model.encode, name, self.token_map, is_first=i == 0))  # type: ignore
        input_ids = torch.cat(input_ids)  # type: ignore
        #query = torch.tensor(query, dtype=torch.long)
        if len(input_ids) > self.max_length:
            tail = input_ids[-1]
            input_ids = input_ids[: self.max_length]
            input_ids[-1] = tail
        token_type_ids = torch.zeros_like(input_ids)
        attention_mask = torch.ones_like(input_ids)
        feature = {
            "product": {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
            },
            "query": {
                "input_ids": query,
                "token_type_ids": torch.zeros_like(query),
                "attention_mask": torch.ones_like(query),
            },
        }
        meta = {
            "product_id": product_id,
            "query_id": query_id,
            "example_id": example_id,
            "pad_token_id": self.cfg.model.pad_token_id,
            "sample_length": self.sample_length[query_id],
        }
        if self.split != "train":
            return feature, esci_label, meta
        else:
            example_id = meta["example_id"]
            kd = []
            for f in self.kd:
                kd.append(
                    torch.tensor(
                        f.loc[example_id][["p0", "p1", "p2", "p3"]].values,
                        dtype=torch.float32,
                    )
                )
            kd = torch.stack(kd)
            return feature, esci_label, meta, kd

    def worker_init(self) -> None:
        self.database = h5py.File(self.filename, "r", libver="latest", swmr=True)
        self.kd = [pd.read_csv(f).set_index("example_id") for f in self.cfg.disk.kd]

        atexit.register(self.cleanup)
            
    @staticmethod
    def collate_fn(batch: List) -> dict:
        features = {}
        pad_token_id = batch[0][2]["pad_token_id"]

        product = {}

        product["input_ids"] = pad_sequence(
            [x[0]["product"]["input_ids"] for x in batch],
            batch_first=True,
            padding_value=pad_token_id,
        )

        product["token_type_ids"] = pad_sequence(
            [x[0]["product"]["token_type_ids"] for x in batch],
            batch_first=True,
        )

        product["attention_mask"] = pad_sequence(
            [x[0]["product"]["attention_mask"] for x in batch],
            batch_first=True,
        )

        query = {}

        query["input_ids"] = pad_sequence(
            [x[0]["query"]["input_ids"] for x in batch],
            batch_first=True,
            padding_value=pad_token_id,
        )
        query["token_type_ids"] = pad_sequence(
            [x[0]["query"]["token_type_ids"] for x in batch],
            batch_first=True,
        )
        query["attention_mask"] = pad_sequence(
            [x[0]["query"]["attention_mask"] for x in batch],
            batch_first=True,
        )

        features = {"product": product, "query": query}

        label = torch.cat([x[1] for x in batch])

        meta = {}
        meta["product_id"] = [x[2]["product_id"] for x in batch]
        meta["example_id"] = [x[2]["example_id"] for x in batch]
        meta["query_id"] = [x[2]["query_id"] for x in batch]
        meta["sample_length"] = torch.tensor(
            [x[2]["sample_length"] for x in batch], dtype=torch.float
        )

        output = {"features": features, "label": label, "meta": meta}
        if len(batch[0]) == 4:
            output["kd"] = torch.stack([x[3] for x in batch])
        return output
