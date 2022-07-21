import hashlib
import os
from collections import defaultdict
from tkinter import X

import h5py
import numpy as np
import pandas as pd
import swifter
from omegaconf import DictConfig

from .clean import get_clean
from .tokenizer import get_tokenizer
from tqdm import tqdm

__all__ = ["prepare_data"]
__key__ = "prepare"
__map__ = {
    "unknown": -1,
    "irrelevant": 0,
    "complement": 1,
    "substitute": 2,
    "exact": 3,
}

def get_prefix(cfg: DictConfig) -> str:
    string = dict(cfg["clean"]).__repr__()  # add clean config
    string += dict(cfg[__key__]).__repr__()  # add dataset config
    string += cfg.model.name  # add dataset config
    string = string.encode()
    string = hashlib.md5(string).hexdigest()
    string = string[:6]
    task = "rank" if cfg.task == "1" else "cls"
    string = f"{task}-{cfg.locale}-{string}"
    return string


def prepare_data(cfg: DictConfig) -> None:
    prefix = get_prefix(cfg)
    os.makedirs(cfg.disk.output_dir, exist_ok=True)
    filename = f"{cfg.disk.output_dir}/{prefix}.h5"
    if not os.path.exists(filename):
        with h5py.File(filename, "w", libver="latest") as ds:
            ds.swmr_mode = True
            prepare_product_catalogue = eval(cfg[__key__]["product_catalogue"])
            product_catalogue_dict = prepare_product_catalogue(cfg)
            group = ds.create_group("product_catalogue")
            for k, v in tqdm(product_catalogue_dict.items()):
                entry = group.create_group(k)
                for col in v.keys():
                    if type(v[col]) != str:
                        entry.create_dataset(
                            col, data=np.array(v[col]).astype(np.int32)
                        )
                    else:
                        entry.create_dataset(col, data=v[col])
            #del product_catalogue_dict

            for dataset in ["train", "test"]:
                fn = eval(cfg[__key__][dataset])
                data = fn(cfg)
                group = ds.create_group(dataset)
                for k, v in tqdm(data.items()):
                    entry = group.create_group(str(k))
                    for col in v.keys():
                        if type(v[col][0]) != str:
                            entry.create_dataset(
                                col, data=np.array(v[col]).astype(np.int32)
                            )
                        else:
                            entry.create_dataset(col, data=v[col])
            

            if cfg['task'] == "1":
                cfg['task'] = '2'
            elif cfg['task'] == '2': 
                cfg['task'] = '1'
            fn = eval(cfg[__key__]["test"])
            data = fn(cfg)
            group = ds.create_group('other')
            for k, v in tqdm(data.items()):
                    entry = group.create_group(str(k))
                    for col in v.keys():
                        if type(v[col][0]) != str:
                            entry.create_dataset(
                                col, data=np.array(v[col]).astype(np.int32)
                            )
                        else:
                            entry.create_dataset(col, data=v[col])
            if cfg['task'] == "1":
                cfg['task'] = '2'
            elif cfg['task'] == '2': 
                cfg['task'] = '1'
            #del data


def prepare_product_catalogue(cfg: DictConfig) -> dict:
    tokenizer = get_tokenizer(cfg)
    product_catalogue = pd.read_csv(
        f"{cfg.disk.data_dir}/{cfg.disk.task_name[cfg.task]}/{cfg.disk.product_catalogue}"
    )
    product_catalogue = product_catalogue.reset_index()
    product_catalogue = product_catalogue.query(
        "product_locale==@cfg.locale"
    ).reset_index(drop=True)
    product_catalogue.fillna('', inplace=True)

    #product_catalogue = product_catalogue[:10] ###

    product_catalogue_dict = defaultdict(dict)
    for col, fn in cfg.clean.product_catalogue.items():
        clean = get_clean(fn)
        clean = clean()
        temp = (
            product_catalogue[col]
            .swifter.allow_dask_on_strings(enable=True)
            .apply(lambda x: clean(x))
        )
        temp = tokenizer(temp.to_list(), truncation=True)
        for idx, product_id in product_catalogue.product_id.iteritems():
            product_catalogue_dict[product_id][col] = temp["input_ids"][idx]  # type: ignore
    
    for _, row in product_catalogue.iterrows():
        product_catalogue_dict[row.product_id]['index'] = [row['index']]

    product_catalogue_dict = dict(product_catalogue_dict)
    return product_catalogue_dict


def _prepare_query(cfg: DictConfig, df: pd.DataFrame) -> dict:
    tokenizer = get_tokenizer(cfg)
    df = df.query("query_locale==@cfg.locale").reset_index(drop=True)
    df.fillna("", inplace=True)
    df_dict = defaultdict(dict)
    clean = get_clean(cfg.clean.query)
    clean = clean()

    temp = df[["query_id", "query"]].drop_duplicates().reset_index(drop=True)
    temp = temp.set_index("query_id")
    temp["query"] = (
        temp["query"]
        .swifter.allow_dask_on_strings(enable=True)
        .apply(lambda x: clean(x))
    )
    query_embedings = tokenizer(temp["query"].to_list(), truncation=True)

    for idx, query_id in enumerate(temp.index):
        df_dict[query_id]["query"] = query_embedings["input_ids"][idx]  # type: ignore
        sub_df = df.query("query_id==@query_id")
        df_dict[query_id]["product_id"] = sub_df["product_id"].to_list()
        df_dict[query_id]["example_id"] = sub_df["example_id"].to_list()
        df_dict[query_id]["esci_label"] = sub_df["esci_label"].to_list()
        df_dict[query_id]["dataset"] = sub_df["dataset"].to_list()
    df_dict = dict(df_dict)
    return df_dict


def _prepare_query_df(df: pd.DataFrame) -> pd.DataFrame:
    df["esci_label"] = df["esci_label"].apply(lambda x: __map__[x])
    if "query_id" not in df.columns:
        temp = df["query"].unique()
        temp = dict(zip(temp, range(len(temp))))
        df["query_id"] = df["query"].apply(lambda x: temp[x])
    if "example_id" not in df.columns:
        df["example_id"] = df.index
    return df


def prepare_train(cfg: DictConfig):
    df = pd.read_csv(
        f"{cfg.disk.data_dir}/{cfg.disk.task_name[cfg.task]}/{cfg.disk.train}"
    )
    df = _prepare_query_df(df)
    return _prepare_query(cfg, df)


def prepare_test(cfg: DictConfig):
    df = pd.read_csv(
        f"{cfg.disk.data_dir}/{cfg.disk.task_name[cfg.task]}/{cfg.disk.test}"
    )
    df["esci_label"] = "unknown"

    task_1 = pd.read_csv(
        f"{cfg.disk.data_dir}/{cfg.disk.task_name['1']}/{cfg.disk.test}"
    )
    task_1['dataset'] = 1
    df = pd.merge(left=df,right=task_1[['query','product_id','query_locale','dataset']],on=['query','product_id','query_locale'],how='left')
    df['dataset'] = df['dataset'].fillna(0)
    df['dataset'] = df['dataset'].astype(int)
    df = _prepare_query_df(df)
    #df = df[:100].reset_index(drop=True) ###
    return _prepare_query(cfg, df)


def prepare_product_catalogue_all(cfg: DictConfig) -> dict:
    cfg = cfg.copy()
    cfg['task'] = '2'
    product_catalogue_dict = prepare_product_catalogue(cfg)
    return product_catalogue_dict

def prepare_train_all_df(cfg:DictConfig):
    x = pd.read_csv(
        f"{cfg.disk.data_dir}/{cfg.disk.task_name['2']}/{cfg.disk.train}"
    )
    x = x.drop("example_id", axis=1)
    y = pd.read_csv(
        f"{cfg.disk.data_dir}/{cfg.disk.task_name['1']}/{cfg.disk.train}"
    )
    y = y.drop("query_id", axis=1)
    df = x.append(y).reset_index(drop=True)
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    task_1 = pd.read_csv(
        f"{cfg.disk.data_dir}/{cfg.disk.task_name['1']}/{cfg.disk.train}"
    )
    task_1['dataset'] = 1
    df = pd.merge(left=df,right=task_1[['query','product_id','query_locale','dataset']],on=['query','product_id','query_locale'],how='left')
    df['dataset'] = df['dataset'].fillna(0)
    df['dataset'] = df['dataset'].astype(int)
    df = _prepare_query_df(df)
    return df

def prepare_train_all(cfg: DictConfig):
    df =  prepare_train_all_df(cfg)   
    #df = df[:100].reset_index(drop=True) ###
    return _prepare_query(cfg, df)