from omegaconf import DictConfig, ListConfig, OmegaConf
from typing import Optional, Union

from regex import F

__ALL__ = ["get_cfg"]
KEY = "config"


def get_filename(path: str) -> str:
    filename = path.split("/")[-1].split(".")[0]
    return filename


def get_cfg(path: Optional[str] = None) -> DictConfig:
    if path is not None:
        cfg = OmegaConf.load(path)
        cfg = OmegaConf.merge(_C, cfg)
        cfg.experiment.name = get_filename(path)
    else:
        cfg = _C.copy()
        cfg.experiment.name = "NA"
    return cfg  # type: ignore


_C = OmegaConf.create()
_C.seed = 2022
_C.locale = "us"
_C.task = "2"
_C.total_fold = 5
_C.fold = 0
_C.checkpoint = ""
_C.train_all = False
_C.val_all = False
_C.export_to_onnx = False
_C.locale_all = False
_C.sample_task_1 = False

# disk configuration
_C.disk = OmegaConf.create()
_C.disk.data_dir = "/data/kdd2022/amazon/data/"
_C.disk.model_dir = "/data/kdd2022/amazon/checkpoint/"
_C.disk.output_dir = "/data/kdd2022/amazon/output/"
_C.disk.submission_dir = "/data/kdd2022/amazon/submission/"
_C.disk.local_dir = "/home/omnisky/output"
_C.disk.product_catalogue = "product_catalogue-v0.3.csv"
_C.disk.train = "train-v0.3.csv"
_C.disk.test = "test_public-v0.3.csv"
_C.disk.kd = ["/data/kdd2022/amazon/submission/distilbart.csv", "/data/kdd2022/amazon/submission/cocolm.csv"]
_C.disk.sample_file = "/data/kdd2022/amazon/output/task_1_query_id.csv"

_C.disk.task_name = {
    "1": "task_1",
    "2": "task_2",
    "3": "task_3",
}

# dataset
_C.dataset = OmegaConf.create()
_C.dataset.version = "Task1Dataset"
_C.dataset.used_col = [
    "product_brand",
    "product_bullet_point",
    "product_color_name",
    "product_description",
    "product_title",
]
_C.dataset.other = False

# clean
_C.clean = OmegaConf.create()
_C.clean.product_catalogue = OmegaConf.create()
_C.clean.product_catalogue.product_title = "QueryClean"
_C.clean.product_catalogue.product_description = "BulletPointClean"
_C.clean.product_catalogue.product_bullet_point = "BulletPointClean"
_C.clean.product_catalogue.product_brand = "QueryClean"
_C.clean.product_catalogue.product_color_name = "QueryClean"
_C.clean.query = "QueryClean"

# data loader
_C.dataloader = OmegaConf.create()
_C.dataloader.version = "BaseDataLoader"
_C.dataloader.num_workers = 2
_C.dataloader.batch_size = OmegaConf.create()
_C.dataloader.batch_size.train = 4
_C.dataloader.batch_size.val = 4
_C.dataloader.batch_size.test = 4
_C.dataloader.override_drop_last = False

# model
_C.model = OmegaConf.create()
_C.model.name = "sentence-transformers/msmarco-MiniLM-L12-cos-v5"
_C.model.max_length = 512
_C.model.num_labels = 4
_C.model.pad_token_id = 0

_C.model.architecture = "CrossEncoder"
_C.model.architecture_args = OmegaConf.create()
_C.model.use_pretrained = False
_C.model.pretrained_path = ""
_C.model.architecture_args.num_text_types = 6
_C.model.architecture_args.extra_feats_num = 1
_C.model.architecture_args.linear_hidden_size = 128
_C.model.architecture_args.pooler_dropout = 0
_C.model.architecture_args.initializer_range = 0.02
_C.model.architecture_args.pooler_hidden_act = "gelu"
_C.model.architecture_args.require_token_id = True

_C.model.encode = OmegaConf.create()
_C.model.encode.query = 1
_C.model.encode.product_title = 2
_C.model.encode.product_brand = 3
_C.model.encode.product_color_name = 4
_C.model.encode.product_bullet_point = 5
_C.model.encode.product_description = 6

_C.prepare = OmegaConf.create()
_C.prepare.product_catalogue = "prepare_product_catalogue"
_C.prepare.train = "prepare_train"
_C.prepare.test = "prepare_test"

_C.tokenizer = OmegaConf.create()
_C.tokenizer.use_fast = False

_C.experiment = OmegaConf.create()
_C.experiment.name = ""
_C.experiment.logger = OmegaConf.create()
_C.experiment.logger.version = "MLFlowLogger"
_C.experiment.logger.tracking_uri = "file:./mlruns"
_C.experiment.logger.experiment_name = "Default"

_C.experiment.saver = OmegaConf.create()
_C.experiment.saver.verbose = False
_C.experiment.saver.filename = "{experiment}-{{step}}-{{score}}"
_C.experiment.saver.monitor = "score"
_C.experiment.saver.save_top_k = -1
_C.experiment.saver.save_weights_only = False
_C.experiment.saver.save_last = True

_C.loss = OmegaConf.create()
_C.loss.version = "nn.CrossEntropyLoss"
_C.loss.kd_weights = [0.6, 0.2, 0.2]
_C.loss.args = OmegaConf.create()

# optimizer
_C.optimizer = OmegaConf.create()
_C.optimizer.version = "transformers.AdamW"
_C.optimizer.weight_decay = 0.01
_C.optimizer.args = OmegaConf.create()
_C.optimizer.args.lr = 2e-5
_C.optimizer.scheduler = OmegaConf.create()
_C.optimizer.scheduler.use = True
_C.optimizer.scheduler.version = "linear_schedule_with_warmup"
_C.optimizer.scheduler.args = OmegaConf.create()
_C.optimizer.scheduler.args.num_warmup_steps = 10000

# lighting
_C.lighting = OmegaConf.create()
_C.lighting.version = "BaseLightingModule"

_C.metric = OmegaConf.create()
_C.metric.version = "F1Score"
_C.metric.args = OmegaConf.create()
_C.metric.args.num_classes = 4
_C.metric.args.average = "micro"

# trainer
_C.trainer = OmegaConf.create()
_C.trainer.gpus = 1
_C.trainer.auto_select_gpus = True
# _C.trainer.strategy = "ddp"
