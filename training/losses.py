import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        dims = (2, 3, 4)
        intersection = torch.sum(pred * target, dim=dims)
        cardinality = torch.sum(pred + target, dim=dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - torch.mean(dice[:, 1:])

class CompositePancreasLoss(nn.Module):
    def __init__(self, num_classes=3, class_weights=(0.1, 1.0, 3.0), dice_weight=0.5, ce_weight=0.5, focal_gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.focal_gamma = focal_gamma
        self.dice_loss = DiceLoss()
        self.weights = torch.tensor(class_weights, dtype=torch.float32)

    def forward(self, seg_logits, loc_heatmap, targets):
        device = seg_logits.device
        weights = self.weights.to(device)
        targets_squeeze = targets.squeeze(1).long()
        ce = F.cross_entropy(seg_logits, targets_squeeze, weight=weights, reduction="none")
        pt = torch.exp(-ce)
        focal_loss = ((1.0 - pt) ** self.focal_gamma * ce).mean()
        pred_softmax = F.softmax(seg_logits, dim=1)
        target_one_hot = F.one_hot(targets_squeeze, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()
        dice_loss = self.dice_loss(pred_softmax, target_one_hot)
        tumor_mask = (targets_squeeze == 2).unsqueeze(1).float()
        bce_loc = F.binary_cross_entropy(loc_heatmap, tumor_mask)
        total_loss = self.ce_weight * focal_loss + self.dice_weight * dice_loss + 0.2 * bce_loc
        return total_loss, {"focal_loss": focal_loss.item(), "dice_loss": dice_loss.item(), "loc_loss": bce_loc.item()}
