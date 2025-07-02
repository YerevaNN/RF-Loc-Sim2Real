import logging
import os

from omegaconf import DictConfig
from tabulate import tabulate

from src.evaluators.rome_eval import RomeEvaluation

log = logging.getLogger(__name__)


def evaluate(config: DictConfig) -> None:
    is_rome = config["is_rome"]
    
    if is_rome:
        indices = sorted(os.listdir(config["prediction_path"]), key=int)
    else:
        indices = [None]
    for data_idx in indices:
        if is_rome:
            prediction_path = os.path.join(config["prediction_path"], data_idx)
        else:
            prediction_path = config["prediction_path"]
        
        log.info(f"\nEvaluating data_idx={data_idx}")
        evaluator = RomeEvaluation(prediction_path=prediction_path)
        
        log.info(f"RMSE: {evaluator.get_rmse()}")
        log.info(
            "\n" + tabulate(
                [evaluator.get_accuracy(config["error_tolerance"])],
                headers=[f"{t}m acc" for t in config["error_tolerance"]]
            )
        )
