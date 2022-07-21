from utils import *
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import argparse
import os
os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'

@rank_zero_only
def print_cfg(cfg) -> None:
    print(cfg)


@rank_zero_only
def make_dir(cfg)-> None:
    os.makedirs(cfg.disk.model_dir, exist_ok=True)


def main(cfg, ckpt_path="")-> None:
    model = get_lighting(cfg,ckpt_path)
    trainer = get_trainer(cfg)
    dataloader = get_dataloader(cfg)
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument(
        "-c",
        "--config",
        default="config/task-2-us-v1.yaml",
        type=str,
    )
    parser.add_argument(
        "--local_rank",
        default=0,
        type=int,
    )
    parser.add_argument(
        "-e",
        "--extra_seed",
        default=0,
        type=int,
    )
    parser.add_argument(
        "-w",
        "--check_point",
        default="",
        type=str,
    )
    args = parser.parse_args()
    cfg = get_cfg(args.config)
    seed_everything(cfg.seed+args.extra_seed)

    prepare_data(cfg)
    make_dir(cfg)
    print_cfg(cfg)
    main(cfg, args.check_point)