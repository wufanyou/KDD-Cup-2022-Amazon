from utils import *
import argparse
import torch
import pandas as pd
from pytorch_lightning import seed_everything
from collections import defaultdict
from tqdm import tqdm

#os.environ["PYTHONWARNINGS"] = "ignore:semaphore_tracker:UserWarning"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        default="config/task-2-us-v1-3090.yaml",
        type=str,
    )

    parser.add_argument(
        "-w",
        "--weight",
        default="last",
        type=str,
    )

    parser.add_argument(
        "-ds",
        "--dataset",
        default="test",
        type=str,
    )

    parser.add_argument(
        "-d",
        "--device",
        default="cuda:0",
        type=str,
    )
    
    parser.add_argument(
        "-t",
        "--task",
        default=2,
        type=int,
    )
    
    
    args = parser.parse_args()
    cfg = get_cfg(args.config)
    cfg.dataloader.batch_size.val = cfg.dataloader.batch_size.test
    #cfg.val_all = False
    cfg.dataloader.override_drop_last = True
    seed_everything(cfg.seed)
    
    if args.task == 1: cfg.dataset.other = True
    model = get_lighting(cfg, args.weight)
    model = model.eval().to(args.device)
    dataloader = get_dataloader(cfg)
    dataloader.setup()
    dataloader = dataloader.get_dataloader(args.dataset)  # type: ignore
    total_output = defaultdict(list)
    with torch.no_grad():
        for input in tqdm(dataloader):
            features = input["features"]
            features = {k: v.to(args.device) for k, v in features.items()}
            output = model(**features)
            output = output.logits.cpu().numpy()
            for i in range(output.shape[1]):
                total_output[f"cls_{i}"] += list(output[:, i])
            for k, v in input["meta"].items():
                total_output[k] += v
            total_output["label"] += list(input["label"].cpu().numpy())
    df = pd.DataFrame(total_output)
    
    if args.task == 1:
        df.to_csv(
            f"{cfg.disk.submission_dir}/task-1-{cfg.experiment.name}-{args.weight.split('/')[-1].split('.')[0]}-{args.dataset}.csv",
            index=False,
        )
    else:
        df.to_csv(
            f"{cfg.disk.submission_dir}/{cfg.experiment.name}-{args.weight.split('/')[-1].split('.')[0]}-{args.dataset}.csv",
            index=False,
        )