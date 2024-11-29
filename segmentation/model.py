import torch.nn as nn
import segmentation_models_pytorch as smp  # type: ignore
from typing import Any


class Model(nn.Module):
    def __init__(self, model_name: str) -> None:
        super(Model, self).__init__()
        self.model_name = model_name.lower()

        self.__models = {
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

        self.__aux_params = dict(
            pooling='avg',
            dropout=0.5,
            activation='sigmoid',
            classes=1,
        )

        self.model = self.__models[model_name](
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            aux_params=self.__aux_params
        )

        if model_name not in self.__models:
            raise ValueError(f"Model not found: {model_name}")

    def forward(self, x) -> Any:
        return self.model(x)
