import logging
import os

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm import tqdm

from src.algorithms.rome_dann import RomeDANN
from src.datamodules.datasets import RomeDataset
from src.datamodules.wair_d_base import DatamoduleBase
from src.utils import EpochCounter

log = logging.getLogger(__name__)


def pred(config: DictConfig) -> None:
    """
    Inference/prediction for DANN models.
    Loads a pretrained DANN checkpoint and runs predictions on the specified dataset split.
    Saves predictions in the same format as inference.py.
    """
    epoch_counter = EpochCounter()
    log.info(f"Instantiating datamodule <{config['datamodule']['_target_']}>")
    datamodule: DatamoduleBase = hydra.utils.instantiate(
        config["datamodule"],
        epoch_counter=epoch_counter, drop_last=False
    )
    datamodule.prepare_data()
    
    log.info(f"Instantiating DANN algorithm {config['algorithm']['_target_']} with checkpoint {config['checkpoint_path']}")
    # Load the DANN model from checkpoint
    algorithm: RomeDANN = hydra.utils.get_class(config["algorithm"]["_target_"]).load_from_checkpoint(
        config["checkpoint_path"], **config["algorithm"],
        network_conf=(OmegaConf.to_yaml(config["network"]) if "network" in config else None),
        gpu=config["gpu"],
        epoch_counter=epoch_counter,
        map_location=f"cuda:{config['gpu']}"
    )
    algorithm.network.eval()
    algorithm.network.cuda(config["gpu"])
    
    # Map rome_idx based on domain parameter
    domain = config.get("domain", "source").lower()
    if domain not in ["source", "target"]:
        raise ValueError(f"Invalid domain '{domain}'. Must be 'source' or 'target'.")
    
    # Source domain: indices 0,1,2 -> 0,1,2 (hard, medium, easy)
    # Target domain: indices 0,1,2 -> 3,4,5 (hard, medium, easy)
    offset = 0 if domain == "source" else 3
    mapped_rome_idx = [idx + offset for idx in config["rome_idx"]]
    log.info(f"Domain: {domain}, Original indices: {config['rome_idx']}, Mapped indices: {mapped_rome_idx}")
    
    pred_path = os.path.dirname(os.path.dirname(config["checkpoint_path"]))
    
    # In case of Rome we have directories in checkpoint path: hard, medium, easy
    if os.path.basename(os.path.dirname(pred_path)) != "outputs":
        pred_path = os.path.dirname(pred_path)
    
    # Add domain suffix to prediction path
    pred_path = os.path.join(
        config["prediction_dir"], 
        f"{os.path.basename(pred_path)}_{domain}"
    )
    os.makedirs(pred_path, exist_ok=True)
    print(f"Prediction path: {pred_path}")
    
    dataset = None
    if config["split"] == "test":
        dataset = datamodule.test_set
    elif config["split"] == "val":
        dataset = datamodule.val_set
    elif config["split"] == "train":
        dataset = datamodule.train_set
    if config["split"] in {"test", "val", "train"}:
        if all(isinstance(d, RomeDataset) for d in dataset):
            for data_idx, data_set in enumerate(dataset):
                if data_idx not in mapped_rome_idx:
                    continue
                # Use original index (0,1,2) for directory naming
                original_idx = data_idx - offset
                log.info(f"Processing {domain} domain, difficulty index={original_idx} (dataset index={data_idx})")
                # noinspection PyTypeChecker
                curr_pred_path = os.path.join(pred_path, str(original_idx))
                os.makedirs(curr_pred_path, exist_ok=True)
                for i in tqdm(range(len(data_set)), total=len(data_set)):
                    batch = data_set[i]
                    out = algorithm.pred(batch)
                    # noinspection PyTypeChecker
                    np.savez(os.path.join(curr_pred_path, f"{i}.npz"), **out)
        else:
            for i in tqdm(range(len(dataset)), total=len(dataset)):
                batch = dataset[i]
                alg_out = algorithm.pred(batch)
                out = alg_out.pop("pred_image")
                out = nn.functional.sigmoid(out)
                out = out.detach().cpu().numpy()[0, 0]
                i = str(i)
                # noinspection PyTypeChecker
                np.savez(os.path.join(pred_path, i), out=out, **alg_out)
    else:
        raise ValueError(f"Unknown split {config['split']}")

