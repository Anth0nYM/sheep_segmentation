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
        self._metric_images: dict = {
            'train': {},
            'val': {},
            'test': {}
        }

    def run_metrics(self,
                    yt: torch.Tensor,
                    yp: torch.Tensor,
                    epoch: int,
                    split: str
                    ):

        self._init_metric_for_epoch(epoch, split)

        self._metric_images[split][epoch].append(
            ImageMetric(
                self._iou(yt, yp),
                self._dice(yt, yp),
                self._current_batch))
        last = self._metric_images[split][epoch][-1]

        return last.iou_batch_mean(), last.dice_batch_mean()

    def _iou(self,
             y_true: torch.Tensor,
             y_pred: torch.Tensor
             ) -> list[float]:

        iou_list = []
        for i in range(self._num_clases):
            y_true_cls = y_true[:, i, :, :]
            y_pred_cls = y_pred[:, i, :, :]

            intersection = torch.sum(y_true_cls * y_pred_cls, dim=(1, 2))
            union = torch.sum(y_true_cls, dim=(1, 2)) + \
                torch.sum(y_pred_cls, dim=(1, 2)) - intersection

            # Prevents zero division
            iou = intersection / torch.clamp(union, min=1e-6)
            iou_list.append(iou.mean().item())  # Mean over batch

        return iou_list

    def _dice(self,
              y_true: torch.Tensor,
              y_pred: torch.Tensor
              ) -> list[float]:

        dice_list = []
        for i in range(self._num_clases):
            y_true_cls = y_true[:, i, :, :]
            y_pred_cls = y_pred[:, i, :, :]

            intersection = torch.sum(y_true_cls * y_pred_cls, dim=(1, 2))
            dice_denominator = torch.sum(y_true_cls, dim=(1, 2)) + \
                torch.sum(y_pred_cls, dim=(1, 2))

            dice = (2 * intersection) / torch.clamp(dice_denominator, min=1e-6)
            dice_list.append(dice.mean().item())

        return dice_list

    def _init_metric_for_epoch(self,
                               epoch: int,
                               split: str
                               ) -> None:

        if epoch not in self._metric_images.keys():
            self._metric_images[split][epoch] = []
            self._current_batch = 0
        else:
            self._current_batch += 1

    def _torch2np(self,
                  tensor: torch.Tensor
                  ) -> np.ndarray:

        return 1. * (tensor.detach().cpu().numpy() > self._tresh)
