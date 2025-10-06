import torch
import torch.nn as nn
from torchvision.models import resnet34 as resnet
from .DeiT import deit_small_patch16_224 as deit
import torch.nn.functional as F

class FAFuse_Classifier(nn.Module):
    """
    FAFuse model adapted for multi-class classification.
    It uses the powerful CNN-Transformer encoder and fusion blocks,
    but replaces the segmentation decoder with a simple classification head.
    """
    def __init__(self, num_classes=7, pretrained=True):
        super(FAFuse_Classifier, self).__init__()

        # --- Load Pre-trained Encoder Components ---
        self.resnet = resnet()
        if pretrained:
            self.resnet.load_state_dict(torch.load('pretrained/resnet34-43635321.pth'), strict=False)
        # We don't need the final layers of ResNet
        self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        self.transformer = deit(pretrained=pretrained)

        # --- Classification Head ---
        # We will take the fused feature map, condense it, and pass it to a linear layer.
        # The fused features from G0 and T0 have 768 channels.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(768, num_classes)

        # --- We still need the fusion block to combine CNN and Transformer features ---
        # Note: The original fusion block is complex. For classification, we can
        # simplify by just concatenating and using a conv layer.
        self.fusion_conv = nn.Conv2d(256 + 768, 768, kernel_size=1)


    def forward(self, imgs):
        # --- Transformer Branch ---
        x_b = self.transformer(imgs)
        x_b = torch.transpose(x_b, 1, 2)
        
        # --- FIX: Changed reshape from (24, 24) to (14, 14) ---
        # This must match the input image size (224 / 16 = 14)
        x_b = x_b.view(x_b.shape[0], -1, 14, 14)

        # --- CNN Branch (only up to layer3 for G0) ---
        x_u = self.resnet.conv1(imgs)
        x_u = self.resnet.bn1(x_u)
        x_u = self.resnet.relu(x_u)
        x_u = self.resnet.maxpool(x_u)
        x_u = self.resnet.layer1(x_u)
        x_u = self.resnet.layer2(x_u)
        x_u = self.resnet.layer3(x_u) # G0 features, shape: [batch, 256, 28, 28]

        # --- Feature Fusion ---
        # We need to make the spatial dimensions of G0 and T0 match before fusing.
        # Upsample T0's feature map to match G0's.
        x_b_upsampled = F.interpolate(x_b, size=x_u.shape[2:], mode='bilinear', align_corners=True)
        
        # Concatenate along the channel dimension
        fused_features = torch.cat([x_u, x_b_upsampled], dim=1)
        fused_features = self.fusion_conv(fused_features) # Resulting shape: [batch, 768, 28, 28]

        # --- Classification Head ---
        # Condense the spatial dimensions into a single vector
        x = self.avgpool(fused_features) # Shape: [batch, 768, 1, 1]
        x = torch.flatten(x, 1) # Shape: [batch, 768]
        
        # Final prediction
        logits = self.fc(x) # Shape: [batch, num_classes]

        return logits
