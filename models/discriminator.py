import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_channels, out_channels, stride=1, normalize=True):
            """Helper function to create discriminator blocks"""
            layers = [
                spectral_norm(
                    nn.Conv2d(in_channels, out_channels, 4, 
                             stride=stride, padding=1, bias=False)
                )
            ]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Main discriminator architecture
        self.model = nn.Sequential(
            # input is (4, 120, 120)
            *discriminator_block(4, 64, stride=2, normalize=False),  # (64, 60, 60)
            *discriminator_block(64, 128, stride=2),                 # (128, 30, 30)
            *discriminator_block(128, 256, stride=2),               # (256, 15, 15)
            *discriminator_block(256, 512, stride=2),               # (512, 8, 8)
            
            # Final classification layer
            nn.Conv2d(512, 1, kernel_size=4, padding=1),           # (1, 7, 7)
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        return self.model(x)

