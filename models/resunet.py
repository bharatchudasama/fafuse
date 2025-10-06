# ResUNet is essentially a U-Net with a ResNet backbone, which our base implementation already uses.
# This file is provided for completeness, but it's architecturally similar to the main U-Net script.
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class FAFuse_ResUNet(nn.Module):
    def __init__(self, pretrained=True):
        super(FAFuse_ResUNet, self).__init__()
        self.cnn_backbone = smp.Unet(
            encoder_name="resnet50", # Using a deeper ResNet backbone
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=1,
        )
        self.cnn_encoder = self.cnn_backbone.encoder

    def forward(self, x):
        features = self.cnn_encoder(x)
        cnn_feat2 = features[2]
        cnn_feat3 = features[3]
        cnn_feat4 = features[4]
        return cnn_feat2, cnn_feat3, cnn_feat4
