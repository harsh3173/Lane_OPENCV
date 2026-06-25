"""Shared device, normalization, losses, and metrics for the VGG-UNet lane detector."""
import numpy as np
import torch
import torch.nn as nn

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def denorm(img_t: torch.Tensor) -> np.ndarray:
    """Tensor (C,H,W) normalized -> HxWxC float image in [0,1]."""
    img = img_t.detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip(img * IMAGENET_STD + IMAGENET_MEAN, 0, 1)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return (1 - num / den).mean()


def make_criterion(device: torch.device, pos_weight: float = 10.0):
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    return lambda logits, target: bce(logits, target) + dice_loss(logits, target)


@torch.no_grad()
def eval_metrics(model, dl, device, thr: float = 0.5):
    """Return (IoU, Dice) over a dataloader."""
    model.eval()
    inter = union = dice_n = dice_d = 0.0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        p = (torch.sigmoid(model(x)) > thr).float()
        inter += (p * y).sum().item()
        union += ((p + y) >= 1).float().sum().item()
        dice_n += 2 * (p * y).sum().item()
        dice_d += (p.sum() + y.sum()).item()
    iou = inter / (union + 1e-6)
    dice = dice_n / (dice_d + 1e-6)
    return iou, dice
