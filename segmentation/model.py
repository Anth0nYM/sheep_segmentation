import torch.nn as nn
import segmentation_models_pytorch as smp  # type: ignore
from typing import Any

# TODO Encorders demais, vou fazer uma pesquisa para saber qual usar


class Model(nn.Module):
    def __init__(self, model_name: str) -> None:
        super(Model, self).__init__()
        self.model_name = model_name.lower()

        models = {
            'unet': smp.Unet,
            'unetplusplus': smp.UnetPlusPlus,
            'fnp': smp.FPN,
            'pspnet': smp.PSPNet,
            'deeplabv3': smp.DeepLabV3,
            'deeplabv3plus': smp.DeepLabV3Plus,
            'linknet': smp.Linknet,
            'manet': smp.MAnet,
            'pan': smp.PAN,
        }

        self.model = models[model_name](
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )

        if model_name not in models:
            raise ValueError(f"Model not found: {model_name}")

    def forward(self, x) -> Any:
        return self.model(x)
