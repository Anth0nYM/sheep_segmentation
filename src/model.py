import torch.nn as nn
import segmentation_models_pytorch as smp


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

        if model_name not in self.__models:
            raise ValueError(f"Model not found: {model_name}")

        self.segmentation_model = self.__models[model_name](
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None
        )

        self.regression_tail = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # (B, C, 1, 1)
            nn.Flatten(),  # (B, C)
            nn.Linear(512, 1)
        )

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        features = self.segmentation_model.encoder(x)
        encoder_output = features[-1]

        seg_output = self.sigmoid(self.segmentation_model(x))
        reg_output = self.relu(self.regression_tail(encoder_output))

        return seg_output, reg_output
