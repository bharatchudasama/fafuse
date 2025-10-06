import torch
import torch.nn as nn
import torch.nn.functional as F
from .DeiT import deit_small_patch16_224

# Helper Functions (as in the original FAFuse.py)
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# --- The Final, Dynamic FAFuse Hybrid Class ---

class FAFuse_Hybrid(nn.Module):
    def __init__(self, cnn_backbone, pretrained=True):
        super(FAFuse_Hybrid, self).__init__()
        
        self.cnn_backbone = cnn_backbone
        self.transformer = deit_small_patch16_224(pretrained=True)
        self.embedding_dim = 768 # DeiT small patch16
        
        # --- Access the encoder directly from the cnn_backbone wrapper ---
        # The wrapper classes in models/ now directly expose the encoder.
        encoder_channels = self.cnn_backbone.encoder.out_channels
        
        # We need the channels from the last three stages for fusion
        cnn_channels_s4 = encoder_channels[-1] # Deepest features
        cnn_channels_s3 = encoder_channels[-2]
        cnn_channels_s2 = encoder_channels[-3]
        
        # Fusion Block for Stage 4
        self.ff4 = FAFusion_block(channel_1=cnn_channels_s4, channel_2=self.embedding_dim) 
        
        # Fusion Block for Stage 3
        self.ff3 = FAFusion_block(channel_1=cnn_channels_s3, channel_2=self.embedding_dim)
        
        # Fusion Block for Stage 2
        self.ff2 = FAFusion_block(channel_1=cnn_channels_s2, channel_2=self.embedding_dim)
        
        # Decoder
        self.conv_out1 = nn.Conv2d(cnn_channels_s4, 1, kernel_size=1, stride=1, padding=0)
        self.conv_out2 = nn.Conv2d(cnn_channels_s3, 1, kernel_size=1, stride=1, padding=0)
        self.conv_out3 = nn.Conv2d(cnn_channels_s2, 1, kernel_size=1, stride=1, padding=0)
        
        # Other layers
        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # ---- Transformer Branch ----
        imgs = F.interpolate(x, size=(352, 352), mode='bilinear', align_corners=True)
        x_b = self.transformer(imgs)
        x_b = x_b.transpose(1, 2).reshape(B, self.embedding_dim, 22, 22)
        
        # ---- CNN Branch (Using our new backbone) ----
        # The wrapper's forward method now directly calls the encoder.
        cnn_features = self.cnn_backbone(x)
        cnn_feat4 = cnn_features[-1] # Deepest features -> Stage 4
        cnn_feat3 = cnn_features[-2] # -> Stage 3
        cnn_feat2 = cnn_features[-3] # -> Stage 2
        
        # ---- Fusion and Decoding ----
        # Stage 4
        fuse_feat4 = self.ff4(cnn_feat4, F.interpolate(x_b, size=cnn_feat4.shape[2:], mode='bilinear', align_corners=True))
        map_4 = self.up_8(self.conv_out1(fuse_feat4))
        
        # Stage 3
        fuse_feat3 = self.ff3(cnn_feat3, F.interpolate(x_b, size=cnn_feat3.shape[2:], mode='bilinear', align_corners=True))
        map_3 = self.up_4(self.conv_out2(fuse_feat3))
        
        # Stage 2
        fuse_feat2 = self.ff2(cnn_feat2, F.interpolate(x_b, size=cnn_feat2.shape[2:], mode='bilinear', align_corners=True))
        map_2 = self.up_2(self.conv_out3(fuse_feat2))
        
        return map_4, map_3, map_2


class FAFusion_block(nn.Module):
    def __init__(self, channel_1, channel_2):
        super(FAFusion_block, self).__init__()
        self.relu = nn.ReLU(True)
        self.ca = ChannelAttention(channel_1)
        self.sa = SpatialAttention()
        self.conv_down1 = conv1x1(channel_1 + channel_2, channel_1)
        self.conv_down2 = conv1x1(channel_1, channel_1)
        self.conv_up = conv1x1(channel_1, channel_1)
        self.conv_final = conv3x3(channel_1, channel_1)

    def forward(self, x1, x2):
        x_f = torch.cat((x1, x2), 1)
        x_f = self.conv_down1(x_f)
        x_f = self.relu(x_f)
        x_f = self.conv_down2(x_f)
        x_ca = self.ca(x1) * x_f
        x_sa = self.sa(x_f) * x_f
        x_csa = x_ca + x_sa
        x_csa = self.conv_up(x_csa)
        x_csa = self.relu(x_csa)
        x_csa = self.conv_final(x_csa)
        return x_csa


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

