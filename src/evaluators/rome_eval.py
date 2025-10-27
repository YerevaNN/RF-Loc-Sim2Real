import os

import numpy as np


class RomeEvaluation:
    
    def __init__(self, prediction_path):
        self.prediction_path = prediction_path
        self.calculated_rmses = None
        self.calculated_sizes = None
    
    @property
    def rmses(self):
        if self.calculated_rmses is None:
            self._load_results()
        return self.calculated_rmses
    
    @property
    def sizes(self):
        if self.calculated_sizes is None:
            self._load_results()
        return self.calculated_sizes
    
    def _load_results(self) -> None:
        rmses: list[float] = []
        sizes: list[float] = []
        for i in sorted(os.listdir(self.prediction_path)):
            pred_path = os.path.join(self.prediction_path, i)
            if not os.path.exists(pred_path):
                break
            
            out = np.load(pred_path, allow_pickle=True)
            image_size = float(out["original_img_size"])  # total map width/height in meters
            ue_loc_y_x = out["ue_loc_y_x"]
            out_img = out["out"]
            
            scale = image_size / out_img.shape[0]
            max_ind = out_img.flatten().argmax()
            # ue_location_pred = np.array([max_ind // max(out_img.shape), max_ind % max(out_img.shape)])
            ue_location_pred = np.array([111.5, 111.5])
            rmse = float(((ue_location_pred - ue_loc_y_x) ** 2).sum() ** 0.5) * scale
            rmses.append(rmse)
            sizes.append(image_size)
        
        self.calculated_rmses = np.array(rmses)
        self.calculated_sizes = np.array(sizes)
        print(self.calculated_rmses.tolist())
        print(self.calculated_sizes.tolist())
    
    def get_rmses(self) -> np.ndarray:
        return self.rmses
    
    def get_rmse(self):
        return self.rmses.mean()
    
    def get_accuracy(self, allowable_errors: int) -> list[float]:
        allowable_errors = np.array(allowable_errors)[np.newaxis]
        return ((self.rmses[:, np.newaxis] < allowable_errors).sum(axis=0) / len(self.rmses)).tolist()
    
    def get_bin_rmse(self, size_bins: list[int]) -> list[list[float]]:
        sizes = self.sizes
        rmses = self.rmses
        rows: list[list[float]] = []
        if len(size_bins) < 2:
            # Not enough edges to form intervals
            return [[str(size_bins[0]) if size_bins else "", int(len(rmses)),
                     float(rmses.mean()) if len(rmses) else float('nan')]]
        sorted_edges = sorted(size_bins)
        for idx in range(len(sorted_edges) - 1):
            lower = sorted_edges[idx]
            upper = sorted_edges[idx + 1]
            if idx < len(sorted_edges) - 2:
                mask = (sizes >= lower) & (sizes < upper)
                label = f"[{int(lower)}, {int(upper)})"
            else:
                # include upper bound for the last interval
                mask = (sizes >= lower) & (sizes <= upper)
                label = f"[{int(lower)}, {int(upper)}]"
            count = int(mask.sum())
            if count == 0:
                rmse_mean = float('nan')
            else:
                rmse_mean = float(rmses[mask].mean())
            rows.append([label, count, rmse_mean])
        return rows
