import torch.nn as nn
import segmentation_models_pytorch as smp

class FAFuse_UNet(nn.Module):
    """
    This is a wrapper class for a U-Net encoder.
    Its only purpose is to create the encoder from the segmentation-models-pytorch library
    and provide a clean, direct interface for the main FAFuse_Hybrid model.
    """
    def __init__(self, pretrained=True):
        super(FAFuse_UNet, self).__init__()
        
        # This class now directly creates and holds the encoder as an attribute.
        # This is the object that the main hybrid model will access.
        self.encoder = smp.encoders.get_encoder(
            name="resnet34",
            in_channels=3,
            depth=5,
            weights="imagenet" if pretrained else None,
        )

    def forward(self, x):
        """
        The forward pass returns the features from the encoder's different stages.
        """
        # We call the encoder's forward method directly.
        features = self.encoder(x)
        return features

