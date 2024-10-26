import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights

class SRGANLoss(nn.Module):
    def __init__(self, device='cpu', lambda_content=20, lambda_perceptual=0.05, 
                 lambda_adv=0.0001, label_smoothing=0.1,lambda_spectral=0.1):
        super(SRGANLoss, self).__init__()
        self.device = device
        self.lambda_content = lambda_content
        self.lambda_perceptual = lambda_perceptual
        self.lambda_adv = lambda_adv
        self.label_smoothing = label_smoothing
        self.lambda_spectral = lambda_spectral

        # VGG for perceptual loss
        self.vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:36].eval().to(device)
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # Add L1 loss for better preservation of spectral information
        self.l1_loss = nn.L1Loss()
        # Add MSE loss for PSNR optimization
        self.mse_loss = nn.MSELoss()
        # Register Sobel filters
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0))

    def edge_loss(self, sr, hr):
        """Calculate edge loss"""
        sr_edges_x = F.conv2d(sr, self.sobel_x.expand(sr.size(1), 1, 3, 3), padding=1, groups=sr.size(1))
        sr_edges_y = F.conv2d(sr, self.sobel_y.expand(sr.size(1), 1, 3, 3), padding=1, groups=sr.size(1))
        hr_edges_x = F.conv2d(hr, self.sobel_x.expand(hr.size(1), 1, 3, 3), padding=1, groups=hr.size(1))
        hr_edges_y = F.conv2d(hr, self.sobel_y.expand(hr.size(1), 1, 3, 3), padding=1, groups=hr.size(1))
        
        return F.l1_loss(sr_edges_x, hr_edges_x) + F.l1_loss(sr_edges_y, hr_edges_y)

    def normalize_for_vgg(self, x):
        """Normalize images for VGG"""
        # Scale to [0,1]
        x = (x + 1) / 2
        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        return (x - mean) / std

    def perceptual_loss(self, sr, hr):
        # Normalize inputs
        sr = self.normalize_for_vgg(sr[:, :3])  # Use first 3 channels
        hr = self.normalize_for_vgg(hr[:, :3])
        
        sr_features = []
        hr_features = []
        
        for i, layer in enumerate(self.vgg.children()):
            sr = layer(sr)
            hr = layer(hr)
            if i in [2, 8, 13, 20]:  # Select specific layers
                sr_features.append(sr)
                hr_features.append(hr)
        
        percep_loss = sum(F.l1_loss(sr_feat, hr_feat) 
                         for sr_feat, hr_feat in zip(sr_features, hr_features))
        
        return percep_loss / len(sr_features)

    def feature_matching_loss(self, real_features, fake_features):
        """Match features between real and fake images"""
        if not isinstance(real_features, list) or not isinstance(fake_features, list):
            return torch.tensor(0.0).to(self.device)
            
        fm_loss = sum(self.feature_matcher(fake_feat, real_feat.detach())
            for real_feat, fake_feat in zip(real_features, fake_features))
        
        return fm_loss / len(real_features)
    
    def total_variation_loss(self, x):
        """Calculate total variation loss for smoothness"""
        h_tv = torch.mean(torch.abs(x[..., 1:, :] - x[..., :-1, :]))
        w_tv = torch.mean(torch.abs(x[..., :, 1:] - x[..., :, :-1]))
        return h_tv + w_tv
    
    def color_consistency_loss(self, sr, hr):
        sr_mean = sr.mean(dim=[2, 3], keepdim=True)
        hr_mean = hr.mean(dim=[2, 3], keepdim=True)
        sr_std = sr.std(dim=[2, 3], keepdim=True)
        hr_std = hr.std(dim=[2, 3], keepdim=True)
        return F.mse_loss(sr_mean, hr_mean) + F.mse_loss(sr_std, hr_std)


    def generator_loss(self, sr, hr, d_sr):
        # Content loss (L1 + MSE)
        l1_loss = self.l1_loss(sr, hr)
        mse_loss = self.mse_loss(sr, hr)
        content_loss = 0.5 * (l1_loss + mse_loss)
        
        # Edge loss
        edge_loss = self.edge_loss(sr, hr)
        
        # Perceptual loss
        percep_loss = self.perceptual_loss(sr, hr)
        
        # Adversarial loss
        target_real = torch.ones_like(d_sr).to(self.device) * (1 - self.label_smoothing)
        adv_loss = F.binary_cross_entropy_with_logits(d_sr, target_real)
        
        # Total variation loss
        tv_loss = self.total_variation_loss(sr)
        
        # Combined loss
        total_loss = (
            self.lambda_content * content_loss +
            self.lambda_perceptual * percep_loss +
            self.lambda_adv * adv_loss +
            self.lambda_spectral * (tv_loss + edge_loss)
        )
        
        return total_loss, {
            'content': content_loss.item(),
            'perceptual': percep_loss.item(),
            'adversarial': adv_loss.item(),
            'edge': edge_loss.item(),
            'total': total_loss.item()
        }


    def discriminator_loss(self, real_pred, fake_pred):
        # Real loss with label smoothing
        real_label = torch.ones_like(real_pred).to(self.device) * (1 - self.label_smoothing)
        real_loss = F.binary_cross_entropy_with_logits(real_pred, real_label)
        
        # Fake loss
        fake_label = torch.zeros_like(fake_pred).to(self.device)
        fake_loss = F.binary_cross_entropy_with_logits(fake_pred, fake_label)
        
        return (real_loss + fake_loss) / 2