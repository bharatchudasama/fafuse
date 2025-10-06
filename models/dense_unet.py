# DenseUNet uses a DenseNet backbone. We can easily configure this.
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class FAFuse_DenseUNet(nn.Module):
    def __init__(self, pretrained=True):
        super(FAFuse_DenseUNet, self).__init__()
        self.cnn_backbone = smp.Unet(
            encoder_name="densenet121", # Using DenseNet as the backbone
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=1,
        )
        self.cnn_encoder = self.cnn_backbone.encoder

    def forward(self, x):
        features = self.cnn_encoder(x)
        # DenseNet features have different indices
        cnn_feat2 = features[2]
        cnn_feat3 = features[3]
        cnn_feat4 = features[4]
        return cnn_feat2, cnn_feat3, cnn_feat4
