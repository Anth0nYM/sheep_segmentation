import numpy as np
import torch


class ImageMetric:
    def __init__(self,
                 ious: list[float],
                 dices: list[float],
                 batch_idx: int
                 ) -> None:

        self._ious = ious
        self._dices = dices
        self._batch_idx = batch_idx

    def __str__(self):
        return (f'Batch: {self._batch_idx} '
                f'IoU: {np.mean(self._ious):.4f} '
                f'Dice: {np.mean(self._dices):.4f}')

    def iou_batch_mean(self, scale: float = 100):
        return scale * np.mean(self._ious)

    def dice_batch_mean(self, scale: float = 100):
        return scale * np.mean(self._dices)


class MetricSegmentation:
    def __init__(self,
                 num_clases: int = 1,
                 tresh: float = 0.5
                 ) -> None:

        self._num_clases = num_clases
        self._tresh = tresh
        self._current_batch = 0
        self._metric_images: dict[str, dict[np.ndarray, float]] = {
            'train': {},
            'val': {},
            'test': {}
        }

    def run_metrics(self,
                    y_true: np.ndarray,
                    y_pred: np.ndarray,
                    epoch: int,
                    split: str
                    ) -> None:
        return None

    def _iou(self,
             y_true: np.ndarray,
             y_pred: np.ndarray
             ) -> list[float]:

        iou_list = []
        for i in range(self._num_clases):
            intersection = np.logical_and(y_true[:, i], y_pred[:, i]).sum()
            union = np.logical_or(y_true[:, i], y_pred[:, i]).sum()
            iou = intersection / union if union != 0 else 1
            iou_list.append(iou)
        return iou_list

    def _dice(self,
              y_true: np.ndarray,
              y_pred: np.ndarray
              ) -> list[float]:

        dice_list = []
        for i in range(self._num_clases):
            intersection = np.logical_and(y_true[:, i], y_pred[:, i]).sum()
            dice_denominator = np.sum(y_true[:, 1]) + np.sum(y_pred[:, 1])
            dice = (2 * intersection) / dice_denominator \
                if dice_denominator != 0 else 1
            dice_list.append(dice)
        return dice_list

    def _init_metric_for_epoch(self) -> None:
        return None

    def _torch2np(self,
                  tensor: torch.Tensor
                  ) -> np.ndarray:

        return tensor.detach().cpu().numpy()
