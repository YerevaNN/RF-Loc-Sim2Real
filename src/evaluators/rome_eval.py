import os

import numpy as np


class RomeEvaluation:
    
    def __init__(self, prediction_path):
        self.prediction_path = prediction_path
        self.calculated_rmses = None
    
    @property
    def rmses(self):
        if self.calculated_rmses is None:
            self.calculated_rmses = self.get_rmses()
        return self.calculated_rmses
    
    def get_rmses(self) -> np.ndarray:
        rmses = []
        for i in sorted(os.listdir(self.prediction_path)):
            pred_path = os.path.join(self.prediction_path, i)
            if not os.path.exists(pred_path):
                break
            
            out = np.load(pred_path, allow_pickle=True)
            image_size = out["original_img_size"]
            ue_loc_y_x = out["ue_loc_y_x"]
            out = out["out"]
            
            scale = image_size / out.shape[0]
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
