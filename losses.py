import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt as edt
import numpy as np

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss','HybridBoundaryLoss']


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss









# 新增加的混合loss

def compute_sdf(target_mask):
    # target_mask: tensor [B, 1, H, W]
    target_np = target_mask.cpu().numpy()
    sdf_list = []
    for b in range(target_np.shape[0]):
        posmask = target_np[b][0].astype(bool)
        negmask = ~posmask
        posdis = edt(negmask)
        negdis = edt(posmask)
        sdf = posdis - negdis
        sdf_list.append(sdf)
    sdf = torch.from_numpy(np.stack(sdf_list)).to(target_mask.device)
    return sdf

class StableBoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        sdf = compute_sdf(target)
        multip = (pred - target).abs() * sdf.abs()
        return multip.mean()

class HybridBoundaryLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.boundary = StableBoundaryLoss()
        self.bce_dice = BCEDiceLoss()

    def forward(self, pred, target):
        loss1 = self.bce_dice(pred, target)
        loss2 = self.boundary(pred, target)
        # loss2 通常比 loss1 大 5~10倍, 所以这里要缩放
        return self.alpha * loss1 + self.beta * 0.1 * loss2