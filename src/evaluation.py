import logging
import os

import hydra
from omegaconf import DictConfig
from tabulate import tabulate

from src.datamodules.rome_data import RomeDatamodule
from src.evaluators.rome_eval import RomeEvaluation
from src.utils import EpochCounter

log = logging.getLogger(__name__)


def evaluate(config: DictConfig) -> None:
    epoch_counter = EpochCounter()
    log.info(f"Instantiating datamodule <{config['datamodule']['_target_']}>")
    datamodule = hydra.utils.instantiate(
        config["datamodule"],
        epoch_counter=epoch_counter, drop_last=False
    )
    datamodule.prepare_data()
    dataset = datamodule.test_set if config["split"] == "test" else datamodule.val_set
    is_rome = isinstance(datamodule, RomeDatamodule)
    
    if is_rome:
        indices = sorted(os.listdir(config["prediction_path"]), key=int)
    else:
        indices = [None]
    for data_idx in indices:
        if is_rome:
            prediction_path = os.path.join(config["prediction_path"], data_idx)
            data_idx = int(data_idx)
            dataset_part = dataset[data_idx]
        else:
            prediction_path = config["prediction_path"]
            dataset_part = dataset
        
        log.info(f"\nEvaluating data_idx={data_idx}")
        evaluator = RomeEvaluation(
            prediction_path=prediction_path, dataset=dataset_part
        )
        
        log.info(f"RMSE: {evaluator.get_rmse()}")
        log.info(
            "\n" + tabulate(
                [evaluator.get_accuracy(config["loc_allowable_errors"])],
                headers=[f"{t}m acc" for t in config["loc_allowable_errors"]]
            )
        )
