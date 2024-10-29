import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from torchvision.models import inception_v3, Inception_V3_Weights
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt  # Added this import

class EvaluationMetrics:
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Initializing metrics on device: {self.device}")
        
        # Initialize inception model for FID
        weights = Inception_V3_Weights.DEFAULT
        self.inception = inception_v3(weights=weights)
        self.inception = self.inception.to(self.device)
        self.inception.eval()
        # Remove last linear layer
        self.inception.fc = nn.Identity()

        # Add MSE loss
        self.mse_loss = nn.MSELoss()

    def calculate_mse(self, sr, hr):
        """Mean Squared Error"""
        return self.mse_loss(sr, hr).item()

    def gaussian(self, window_size, sigma):
        """Generate 1D Gaussian kernel"""
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) 
                            for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        """Create SSIM window"""
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def calculate_psnr(self, sr, hr):
        """Peak Signal-to-Noise Ratio with MSE"""
        mse = self.calculate_mse(sr, hr)
        if mse == 0:
            return float('inf')
        # Assuming pixel values are in [-1, 1], so max_pixel_value = 2
        return 20 * torch.log10(torch.tensor(2.0)) - 10 * torch.log10(torch.tensor(mse))

    def calculate_ssim(self, sr, hr, window_size=11):
        """Structural Similarity Index"""
        # Ensure the input is in the correct range [-1, 1]
        channel = sr.size(1)
        window = self.create_window(window_size, channel).to(self.device)
        
        mu1 = F.conv2d(sr, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(hr, window, padding=window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(sr * sr, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(hr * hr, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(sr * hr, window, padding=window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean().item()

    def calculate_fid(self, sr, hr):
        """Fréchet Inception Distance"""
        # Adapt images for inception
        if sr.size(1) == 4:  # If 4 channels, take first 3
            sr = sr[:, :3]
            hr = hr[:, :3]
        sr = (sr + 1) / 2  # Convert from [-1, 1] to [0, 1]
        hr = (hr + 1) / 2
        
        # Get inception features
        with torch.no_grad():
            sr_features = self.inception(sr).cpu().numpy()
            hr_features = self.inception(hr).cpu().numpy()
        
        # Calculate mean and covariance
        mu1 = np.mean(sr_features, axis=0)
        sigma1 = np.cov(sr_features, rowvar=False)
        
        mu2 = np.mean(hr_features, axis=0)
        sigma2 = np.cov(hr_features, rowvar=False)
        
        # Calculate FID
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return float(fid)

    def evaluate_batch(self, sr, hr):
        """Evaluate all metrics for a batch"""
        with torch.no_grad():
            metrics = {
                'MSE': self.calculate_mse(sr, hr),
                'PSNR': self.calculate_psnr(sr, hr).item(),
                'SSIM': self.calculate_ssim(sr, hr),
                'FID': self.calculate_fid(sr, hr)
            }
        return metrics

    def print_metrics(self, metrics, prefix=""):
        """Pretty print metrics with proper formatting"""
        print(f"\n{prefix}Metrics:")
        print(f"MSE: {metrics['MSE']:.4f}")
        print(f"PSNR: {metrics['PSNR']:.4f} dB")
        print(f"SSIM: {metrics['SSIM']:.4f}")
        print(f"FID: {metrics['FID']:.4f}")

    def visualize_batch_comparison(self, sr, hr, save_path=None):
        """Visualize and optionally save comparison between SR and HR images"""
        # Convert tensors to numpy arrays and move to CPU
        sr_np = sr.cpu().numpy()
        hr_np = hr.cpu().numpy()
        
        # Take first image from batch and first 3 channels
        sr_img = sr_np[0, :3].transpose(1, 2, 0)
        hr_img = hr_np[0, :3].transpose(1, 2, 0)
        
        # Normalize to [0, 1] for visualization
        sr_img = (sr_img + 1) / 2
        hr_img = (hr_img + 1) / 2
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot images
        ax1.imshow(sr_img)
        ax1.set_title('Super-Resolution')
        ax1.axis('off')
        
        ax2.imshow(hr_img)
        ax2.set_title('High-Resolution (Ground Truth)')
        ax2.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()