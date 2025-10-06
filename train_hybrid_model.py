import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import argparse
from datetime import datetime
import os
import numpy as np

# --- Import all our new models ---
from models.unet import FAFuse_UNet
from models.unet_plusplus import FAFuse_UNetPlusPlus
from models.attention_unet import FAFuse_AttentionUNet
from models.resunet import FAFuse_ResUNet
from models.dense_unet import FAFuse_DenseUNet

# --- Import the rest of the FAFuse framework and training components ---
from lib.FAFuse_hybrid import FAFuse_Hybrid
from utils.dataloader import get_loader, test_dataset
from utils.utils import AvgMeter

# --- Helper functions from the original training script ---
def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

def get_model(model_name, pretrained):
    """Factory function to select and return the chosen CNN backbone."""
    if model_name == 'unet':
        return FAFuse_UNet(pretrained=pretrained)
    elif model_name == 'unet++':
        return FAFuse_UNetPlusPlus(pretrained=pretrained)
    elif model_name == 'attention_unet':
        return FAFuse_AttentionUNet(pretrained=pretrained)
    elif model_name == 'resunet':
        return FAFuse_ResUNet(pretrained=pretrained)
    elif model_name == 'denseunet':
        return FAFuse_DenseUNet(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def validate(model, path):
    """Validation function to evaluate the model on the test set after each epoch."""
    model.eval()
    test_loader = test_dataset(image_root=f'{path}/data_test.npy', gt_root=f'{path}/mask_test.npy')
    dice_bank = []
    iou_bank = []
    
    with torch.no_grad():
        for i in range(test_loader.size):
            image, gt = test_loader.load_data()
            image = image.cuda()
            gt = 1*(gt>0.5)

            _, _, res = model(image)
            res = F.upsample(res, size=gt.shape[-2:], mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = 1*(res > 0.5)
            
            dice = (2 * (res * gt).sum()) / (res.sum() + gt.sum() + 1e-8)
            iou = (res * gt).sum() / (res.sum() + gt.sum() - (res * gt).sum() + 1e-8)
            dice_bank.append(dice)
            iou_bank.append(iou)

    mean_dice = np.mean(dice_bank) if dice_bank else 0
    mean_iou = np.mean(iou_bank) if iou_bank else 0
    
    print(f'✅ Validation finished! Current Score (Dice+IoU): {mean_dice + mean_iou:.4f} (Dice: {mean_dice:.4f}, IoU: {mean_iou:.4f})')
    return mean_dice + mean_iou

def train(train_loader, model, optimizer, epoch, total_step, opt):
    """Main training loop for one epoch."""
    model.train()
    loss_record2, loss_record3, loss_record4 = AvgMeter(), AvgMeter(), AvgMeter()
    
    for i, pack in enumerate(train_loader, start=1):
        images, gts = pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()

        lateral_map_4, lateral_map_3, lateral_map_2 = model(images)

        gts_size = gts.size()[2:]
        lateral_map_4 = F.interpolate(lateral_map_4, size=gts_size, mode='bilinear', align_corners=True)
        lateral_map_3 = F.interpolate(lateral_map_3, size=gts_size, mode='bilinear', align_corners=True)
        lateral_map_2 = F.interpolate(lateral_map_2, size=gts_size, mode='bilinear', align_corners=True)
        
        loss4 = structure_loss(lateral_map_4, gts)
        loss3 = structure_loss(lateral_map_3, gts)
        loss2 = structure_loss(lateral_map_2, gts)
        loss = 0.2*loss2 + 0.1*loss3 + 0.7*loss4

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        optimizer.zero_grad()

        # --- THIS IS THE FIX ---
        # We remove the .item() call to pass the raw Tensor to the AvgMeter.
        loss_record2.update(loss2, opt.batchsize)
        loss_record3.update(loss3, opt.batchsize)
        loss_record4.update(loss4, opt.batchsize)

        if i % 20 == 0 or i == total_step:
            print(f'{datetime.now()} Epoch [{epoch:03d}/{opt.epoch:03d}], Step [{i:04d}/{total_step:04d}], '
                  f'[L2: {loss_record2.show():.4f}, L3: {loss_record3.show():.4f}, L4: {loss_record4.show():.4f}]')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='unet',
                        choices=['unet', 'unet++', 'attention_unet', 'resunet', 'denseunet'],
                        help='Choose the CNN backbone for the hybrid FAFuse model.')
    parser.add_argument('--epoch', type=int, default=30, help='epoch number')
    parser.add_argument('--lr', type=float, default=7e-5, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
    parser.add_argument('--train_path', type=str, default='data/', help='path to train dataset')
    parser.add_argument('--test_path', type=str, default='data/', help='path to test dataset')
    parser.add_argument('--train_save', type=str, default='FAFuse_Hybrid')
    parser.add_argument('--patience', type=int, default=10, help='Epochs to wait for improvement before early stopping.')
    opt = parser.parse_args()

    print(f"--- Building CNN backbone: {opt.model_name} ---")
    cnn_backbone = get_model(opt.model_name, pretrained=True)

    print("--- Building Full Hybrid FAFuse Model ---")
    model = FAFuse_Hybrid(cnn_backbone=cnn_backbone, pretrained=True).cuda()

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    image_root = f'{opt.train_path}/data_train.npy'
    gt_root = f'{opt.train_path}/mask_train.npy'
    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize)
    total_step = len(train_loader)

    print("\n" + "#"*20, "Start Training", "#"*20 + "\n")

    best_score = 0.0
    epochs_no_improve = 0
    
    for epoch in range(1, opt.epoch + 1):
        print(f"--- Starting Epoch {epoch}/{opt.epoch} ---")
        train(train_loader, model, optimizer, epoch, total_step, opt)
        
        current_score = validate(model, opt.test_path)
        
        if current_score > best_score:
            best_score = current_score
            epochs_no_improve = 0
            save_path = f'snapshots/{opt.train_save}_{opt.model_name}/'
            os.makedirs(save_path, exist_ok=True)
            model_path = os.path.join(save_path, f'best_model_epoch_{epoch}.pth')
            torch.save(model.state_dict(), model_path)
            print(f"🎯 New best score: {best_score:.4f}. Saved model to {model_path}")
        else:
            epochs_no_improve += 1
            print(f"❌ No improvement for {epochs_no_improve} epoch(s). Best score remains: {best_score:.4f}")

        if epochs_no_improve >= opt.patience:
            print(f"\n✋ Early stopping triggered after {opt.patience} epochs with no improvement.")
            break



# ```
# ```

# ---

# ### How the Loss is Calculated

# Here is a detailed explanation of the sophisticated loss calculation used in this training script. It combines two main ideas: a **Structure-aware Loss** and **Deep Supervision**.

# #### 1. The `structure_loss` Function

# This is the core function that evaluates how "wrong" a single prediction is. It's specifically designed to be good at segmentation by combining two different perspectives:

# * **Weighted Binary Cross-Entropy (WBCE):** This is a pixel-level loss. For every pixel, it checks if the model's prediction was correct (lesion or background). Crucially, it uses a **weight map (`weit`)** that gives 5 times more importance to the pixels on the **boundary** of the lesion. This forces the model to work much harder to get the difficult edges right.
# * **Weighted Intersection over Union (WIoU):** This is a shape-level loss. It measures the overall **overlap** between the predicted shape and the true shape. It also uses the same `weit` map, meaning a good overlap on the boundaries is rewarded more heavily.

# By adding these two together (`wbce + wiou`), the model is trained to be both pixel-accurate and structurally correct, with a special focus on getting the outlines perfect.

# #### 2. Deep Supervision

# The FAFuse model produces three separate output masks (`lateral_map_2`, `lateral_map_3`, `lateral_map_4`) at different stages of its decoder. The final loss that the model learns from is a **weighted average** of the `structure_loss` calculated on each of these three maps.

# ```python
# loss2 = structure_loss(lateral_map_2, gts)
# loss3 = structure_loss(lateral_map_3, gts)
# loss4 = structure_loss(lateral_map_4, gts)

# # The final, deepest map gets the most weight (70%)
# loss = 0.2*loss2 + 0.1*loss3 + 0.7*loss4

