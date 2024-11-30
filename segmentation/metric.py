import torch
import segmentation_models_pytorch as smp  # type: ignore
from typing import Optional


class MetricSegmentation:
    def __init__(self,
                 threshold: float = 0.5,
                 reduction: Optional[str] = 'macro'
                 ) -> None:

        self.__threshold = threshold
        self.__reduction = reduction
        self.metric_functions = {
            "IoU": smp.metrics.iou_score,
            "F1 Score": smp.metrics.f1_score,
            "Accuracy": smp.metrics.accuracy,
            "Precision": smp.metrics.precision,
            "Recall": smp.metrics.recall,
            "Sensitivity": smp.metrics.sensitivity,
            "Specificity": smp.metrics.specificity,
            "Balanced Accuracy": smp.metrics.balanced_accuracy,
            "PPV": smp.metrics.positive_predictive_value,
            "NPV": smp.metrics.negative_predictive_value,
            "False Negative Rate (FNR)": smp.metrics.false_negative_rate,
            "False Positive Rate (FPR)": smp.metrics.false_positive_rate,
            "False Discovery Rate (FDR)": smp.metrics.false_discovery_rate,
            "False Omission Rate (FOR)": smp.metrics.false_omission_rate,
            "LR+": smp.metrics.positive_likelihood_ratio,
            "LR-": smp.metrics.negative_likelihood_ratio,
        }

    def run_metrics(self,
                    yt: torch.Tensor,
                    yp: torch.Tensor
                    ) -> dict:

        yp_bin = (yp > self.__threshold).int()
        tp, fp, fn, tn = smp.metrics.get_stats(
            yp_bin, yt.long(), mode="binary", threshold=None
        )

        metrics = {
            name: func(tp, fp, fn, tn, reduction=self.__reduction).item()
            for name, func in self.metric_functions.items()
        }

        return metrics
