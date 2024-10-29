import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ResidualBlock, UpsampleBlock

class Generator(nn.Module):
    def __init__(self, n_res_blocks=16):
        super(Generator, self).__init__()
        
        # Initial feature extraction
        self.conv1_20m = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        
        self.conv1_60m = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        
        # Residual blocks
        self.res_blocks_20m = nn.ModuleList([ResidualBlock(64) for _ in range(n_res_blocks)])
        self.res_blocks_60m = nn.ModuleList([ResidualBlock(64) for _ in range(n_res_blocks)])

        # 20m path (60x60 -> 120x120): 2x upsampling
        self.upsample_20m = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  # Add BatchNorm for stability
            nn.PixelShuffle(2),   # 2x upsampling: 60x60 -> 120x120
            nn.PReLU()
        )
        
        # 60m path (20x20 -> 120x120): 6x upsampling in steps
        # First step: 20x20 -> 40x40
        self.upsample_60m_1 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        
        # Second step: 40x40 -> 80x80
        self.upsample_60m_2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        
        # Final refinement: 80x80 -> 120x120
        self.upsample_60m_3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            # Add an extra conv layer for better feature refinement
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )

        # Fusion layer with increased capacity
        self.fusion = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 4, kernel_size=3, padding=1)
        )

    def forward(self, x_20m, x_60m):
        # Add residual scaling factor
        residual_scale = 0.1
        # Initial features
        feat_20m = self.conv1_20m(x_20m)
        feat_60m = self.conv1_60m(x_60m)
        
        # Residual blocks with skip connections
        res_20m = feat_20m
        res_60m = feat_60m
        
        for res_block_20m, res_block_60m in zip(self.res_blocks_20m, self.res_blocks_60m):
            res_20m = res_20m + residual_scale * res_block_20m(res_20m)
            res_60m = res_60m + residual_scale * res_block_60m(res_60m)
        
        # 20m upsampling path (60x60 -> 120x120)
        up_20m = self.upsample_20m(res_20m)  # 120x120
        
        # 60m upsampling path (20x20 -> 120x120)
        up_60m = self.upsample_60m_1(res_60m)  # 40x40
        up_60m = self.upsample_60m_2(up_60m)   # 80x80
        # Final resize and refinement: 80x80 -> 120x120
        up_60m = F.interpolate(
            up_60m, 
            size=(120, 120), 
            mode='bicubic',  # Changed to bicubic for better quality
            align_corners=True
        )
        up_60m = self.upsample_60m_3(up_60m)
        
        # Fusion
        feat = torch.cat([up_20m, up_60m], dim=1)
        out = self.fusion(feat)
        
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