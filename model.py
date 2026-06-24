"""VGG16-BN encoder + UNet decoder for binary lane segmentation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.models import vgg16_bn, VGG16_BN_Weights


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class VGGUNet(nn.Module):
    """VGG16-BN feature blocks (split at max-pool boundaries) feed UNet skip connections."""

    def __init__(self, pretrained=True, checkpoint_enc=True):
        super().__init__()
        weights = VGG16_BN_Weights.DEFAULT if pretrained else None
        feats = vgg16_bn(weights=weights).features
        self.enc1 = feats[0:6]    # 64,  /1
        self.enc2 = feats[6:13]   # 128, /2
        self.enc3 = feats[13:23]  # 256, /4
        self.enc4 = feats[23:33]  # 512, /8
        self.enc5 = feats[33:43]  # 512, /16 (bottleneck)

        self.dec4 = DecoderBlock(512, 512, 256)
        self.dec3 = DecoderBlock(256, 256, 128)
        self.dec2 = DecoderBlock(128, 128, 64)
        self.dec1 = DecoderBlock(64, 64, 32)
        self.head = nn.Conv2d(32, 1, kernel_size=1)
        # Recompute encoder activations in backward instead of storing them.
        # Trades a little extra compute for a large drop in peak memory.
        self.checkpoint_enc = checkpoint_enc

    def _enc(self, block, x):
        if self.checkpoint_enc and self.training:
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(self, x):
        s1 = self._enc(self.enc1, x)
        s2 = self._enc(self.enc2, s1)
        s3 = self._enc(self.enc3, s2)
        s4 = self._enc(self.enc4, s3)
        b = self._enc(self.enc5, s4)
        d = self.dec4(b, s4)
        d = self.dec3(d, s3)
        d = self.dec2(d, s2)
        d = self.dec1(d, s1)
        return self.head(d)
