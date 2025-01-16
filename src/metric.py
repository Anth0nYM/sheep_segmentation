import torch
import torchmetrics.functional as m
from typing import Callable


class MetricsReport:
    def __init__(self,
                 task: str,
                 threshold: float = 0.5,
                 ) -> None:

        self.__task = task.lower()
        self.__threshold = threshold
        self.metric_functions: dict[str, Callable] = {}

        if self.__task == "segmentation":
            self.metric_functions = {
                "IoU": self.iou,
                "Dice": m.dice
            }
        elif self.__task == "regression":
            self.metric_functions = {
                "MAE": m.mean_absolute_error,
                "RMSE": m.mean_squared_error,
            }
        else:
            raise ValueError("task must be 'segmentation' or 'regression'")

    def run_metrics(self,
                    yt: torch.Tensor,
                    yp: torch.Tensor
                    ) -> dict:

        if self.__task == "segmentation":
            yp_bin = (yp > self.__threshold).float()
            yt = yt.int()

            metrics = {
                name: func(yp_bin, yt).item()
                for name, func in self.metric_functions.items()
            }

        elif self.__task == "regression":
            metrics = {
                name: func(yp, yt).item()
                for name, func in self.metric_functions.items()
            }

        return metrics

    def iou(self, yp: torch.Tensor, yt: torch.Tensor) -> torch.Tensor:
        yp_bin = (yp > self.__threshold).float()
        yt = yt.int()
        return m.jaccard_index(yp_bin, yt, task="binary")
