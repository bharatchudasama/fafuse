import torch
import torch.nn as nn
from functools import partial

from .vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
import torch.nn.functional as F
import numpy as np


__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384',
]


class DeiT(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The positional embedding parameter that will hold the resized weights
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))

    def forward(self, x):
        x = self.patch_embed(x)
        pe = self.pos_embed
        x = x + pe
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x


@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    # --- THIS IS THE FIX ---
    # Initialize the model with the CORRECT image size (352x352) for the segmentation task.
    # This ensures self.pos_embed is created with the right shape (22*22 = 484).
    model = DeiT(
        patch_size=16, embed_dim=768, depth=8, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), img_size=352, **kwargs) # Changed from 384 to 352
    
    model.default_cfg = _cfg()
    if pretrained:
        # Load the DeiT model pre-trained on 384x384 images.
        ckpt = torch.load('pretrained/deit_base_patch16_384-8de9b5d1.pth')
        
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in ckpt['model'].items() if k in model_dict and 'head' not in k}

        # Specifically handle the positional embedding
        pos_embed_pretrained = ckpt['model']['pos_embed']
        pos_embed_new = model_dict['pos_embed']
        
        if pos_embed_new.shape != pos_embed_pretrained.shape:
            # Exclude the first token (class token) from the pretrained embedding
            pe_pretrained = pos_embed_pretrained[:, 1:, :] 
            pe_pretrained = pe_pretrained.transpose(-1, -2)
            # The pretrained embeddings are for a 24x24 grid (from 384x384 image)
            pe_pretrained = pe_pretrained.view(pe_pretrained.shape[0], pe_pretrained.shape[1], 24, 24)
            
            # Resize to match the 22x22 patch grid of our 352x352 input images.
            pe_resized = F.interpolate(pe_pretrained, size=(22, 22), mode='bilinear', align_corners=True)
            
            pe_resized = pe_resized.flatten(2)
            pe_resized = pe_resized.transpose(-1, -2)
            pretrained_dict['pos_embed'] = pe_resized
            
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("Pretrained weights loaded and positional embeddings resized.")

    model.head = nn.Identity()
    return model

