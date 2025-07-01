import os

import numpy as np
from tqdm import tqdm

from src.datamodules.datasets import RomeDataset, WAIRDDatasetPathLoss


class RomeEvaluation:
    
    def __init__(self, prediction_path, dataset):
        assert isinstance(dataset, (RomeDataset, WAIRDDatasetPathLoss)), \
        "RomeEvaluation works only with RomeDataset or WAIRDDatasetPathLoss"

        self.prediction_path = prediction_path
        self.dataset = dataset
        self.calculated_rmses = None
        
    @property
    def rmses(self):
        if self.calculated_rmses is None:
            self.calculated_rmses = self.get_rmses()
        return self.calculated_rmses
    
    def get_rmses(self) -> np.ndarray:
        rmses = []
        indices = range(len(self.dataset))
        for i in tqdm(indices):
            batch = self.dataset[i]
            pred_path = os.path.join(self.prediction_path, f"{i}.npz")
            if not os.path.exists(pred_path):
                break
            
            out = list(np.load(pred_path, allow_pickle=True).values())[0]
            
            if len(batch) > 5:
                input_image, sequence, supervision_image, image_size, ue_loc_y_x, map_center, ue_initial_lat_lon = batch
                scale = image_size / out.shape[0]
            else:
                map_resized, base_stations_data, ue_loc_img, orig_image_size, ue_loc_y_x = batch
                scale = orig_image_size / out.shape[0]
            
            max_ind = out.flatten().argmax()
            ue_location_pred = np.array([max_ind // max(out.shape), max_ind % max(out.shape)])
            rmse = float(((ue_location_pred - ue_loc_y_x) ** 2).sum() ** 0.5) * scale
            rmses.append(rmse)
        
        rmses = np.array(rmses)
        return rmses
    
    def get_rmse(self):
        return self.rmses.mean()
    
    def get_accuracy(self, allowable_errors: int) -> list[float]:
        allowable_errors = np.array(allowable_errors)[np.newaxis]
        return ((self.rmses[:, np.newaxis] < allowable_errors).sum(axis=0) / len(self.rmses)).tolist()
