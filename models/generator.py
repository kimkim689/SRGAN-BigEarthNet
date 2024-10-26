import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ResidualBlock, SELayer

class PixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor):
        super().__init__()
        self.downscale_factor = downscale_factor
        
    def forward(self, x):
        """Rearrange pixels into blocks."""
        b, c, h, w = x.shape
        ds = self.downscale_factor
        return x.view(b, c, h//ds, ds, w//ds, ds).permute(0,1,3,5,2,4).reshape(b, c*ds*ds, h//ds, w//ds)

class SmoothUpsampling(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * scale_factor * scale_factor, 3, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(in_channels * scale_factor * scale_factor),
            nn.PixelShuffle(scale_factor)
        )

    def forward(self, x):
        return self.conv(x)

class Generator(nn.Module):
    def __init__(self, n_res_blocks=16):
        super().__init__()
        
        # Initial convolution layers
        self.conv1_20m = nn.Sequential(
            nn.Conv2d(6, 64, 3, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(64)
        )
        
        self.conv1_60m = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(64)
        )
        
        # Residual blocks
        self.res_blocks_20m = nn.ModuleList([ResidualBlock(64) for _ in range(n_res_blocks)])
        self.res_blocks_60m = nn.ModuleList([ResidualBlock(64) for _ in range(n_res_blocks)])
        
        # Smooth upsampling blocks
        self.upsample_20m = SmoothUpsampling(64, 2)  # 60x60 -> 120x120
        
        # Progressive upsampling for 60m
        self.upsample_60m = nn.Sequential(
            SmoothUpsampling(64, 2),  # 20x20 -> 40x40
            SmoothUpsampling(64, 2),  # 40x40 -> 80x80
            nn.Conv2d(64, 64, 3, padding=1),
            nn.PReLU(),
            nn.Upsample(scale_factor=1.5, mode='bicubic', align_corners=True)  # 80x80 -> 120x120
        )
        
        # Final layers
        self.fusion = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 4, 3, padding=1)
        )

    def forward(self, x_20m, x_60m):
        # Initial features
        feat_20m = self.conv1_20m(x_20m)
        feat_60m = self.conv1_60m(x_60m)
        
        # Residual blocks
        res_20m = feat_20m
        res_60m = feat_60m
        
        for res_block_20m, res_block_60m in zip(self.res_blocks_20m, self.res_blocks_60m):
            res_20m = res_20m + res_block_20m(res_20m)
            res_60m = res_60m + res_block_60m(res_60m)
        
        # Upsampling
        up_20m = self.upsample_20m(res_20m)
        up_60m = self.upsample_60m(res_60m)
        
        # Fusion and final output
        out = self.fusion(torch.cat([up_20m, up_60m], dim=1))
        return torch.tanh(out)


def _initialize_weights(self):
        """Initialize network weights for better training"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)