import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_channels, out_channels, stride=1, normalize=True):
            layers = [
                spectral_norm(nn.Conv2d(in_channels, out_channels, 4, stride=stride, padding=1, bias=False))
            ]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
            
        self.model = nn.Sequential(
            # No normalization for first layer
            *discriminator_block(4, 64, stride=2, normalize=False),  # 60x60
            *discriminator_block(64, 128, stride=2),                 # 30x30
            *discriminator_block(128, 256, stride=2),               # 15x15
            *discriminator_block(256, 512, stride=2),               # 8x8
            nn.Conv2d(512, 1, 4, padding=1)                        # 8x8
        )

    def forward(self, x):
        return self.model(x)