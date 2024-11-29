import torch
from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score

DEVICE = torch.device(device='cuda') if torch.cuda.is_available() else \
        torch.device(device='cpu')


class MetricSegmentation:
    def __init__(self,
                 threshold: float = 0.5
                 ) -> None:

        self._threshold = threshold
        self._iou_metric = BinaryJaccardIndex(threshold=threshold).to(
            device=DEVICE
        )
        self._dice_metric = BinaryF1Score(threshold=threshold).to(
            device=DEVICE
        )

    def run_metrics(self,
                    yt: torch.Tensor,
                    yp: torch.Tensor
                    ):

        # Binarize
        yp_bin = (yp > self._threshold).float()

        iou = self._iou_metric(yp_bin, yt)
        dice = self._dice_metric(yp_bin, yt)

        return iou.item(), dice.item()
