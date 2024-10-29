import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights

class SRGANLoss(nn.Module):
    def __init__(self, device='cpu', lambda_content=1.0, lambda_perceptual=0.1, 
                 lambda_adversarial=0.001, label_smoothing=0.1):
        super().__init__()
        self.device = device
        self.lambda_content = lambda_content
        self.lambda_perceptual = lambda_perceptual
        self.lambda_adversarial = lambda_adversarial
        self.label_smoothing = label_smoothing
        
        # VGG for perceptual loss
        self.vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:36].eval().to(device)
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def perceptual_loss(self, sr, hr):
        """Improved perceptual loss calculation"""
        # Ensure inputs are in [0, 1] range for VGG
        sr = (sr + 1) * 0.5
        hr = (hr + 1) * 0.5
        
        # Clamp to ensure valid range
        sr = torch.clamp(sr, 0, 1)
        hr = torch.clamp(hr, 0, 1)
        
        sr_features = self.vgg(sr[:, :3])
        hr_features = self.vgg(hr[:, :3])
        
        return F.mse_loss(sr_features, hr_features)
    
    def generator_loss(self, sr, hr, d_sr):
        # Content loss (L1 + MSE)
        l1_loss = F.l1_loss(sr, hr)
        mse_loss = F.mse_loss(sr, hr)
        content_loss = 0.5 * (l1_loss + mse_loss)
        
        # Perceptual loss with better normalization
        percep_loss = self.perceptual_loss(sr, hr)
        
        # Adversarial loss with improved stability
        target_real = torch.ones_like(d_sr).to(self.device) * 0.9
        adv_loss = F.binary_cross_entropy_with_logits(d_sr, target_real)
        
        # Combined loss with better balancing
        total_loss = (
            self.lambda_content * content_loss +
            self.lambda_perceptual * percep_loss +
            self.lambda_adversarial * adv_loss
        )
        
        return total_loss, {
            'content': content_loss.item(),
            'perceptual': percep_loss.item(),
            'adversarial': adv_loss.item(),
            'total': total_loss.item()
        }
    def pretrain_generator(self, train_loader, num_epochs=5):
        """Pretrain generator with only content loss"""
        print("Pre-training generator...")

        for epoch in range(num_epochs):
            self.generator.train()
            total_loss = 0

            for batch in tqdm(train_loader):
                if batch is None:
                    continue

                x_20m = batch['bands_20m'].to(self.device)
                x_60m = batch['bands_60m'].to(self.device)
                real_hr = batch['bands_10m'].to(self.device)

                self.g_optimizer.zero_grad()

                fake_hr = self.generator(x_20m, x_60m)
                loss = F.mse_loss(fake_hr, real_hr)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.generator.parameters(), 
                    self.config.gradient_clip
                )
                self.g_optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Pre-training Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    def discriminator_loss(self, real_pred, fake_pred):
        # Real loss
        target_real = torch.ones_like(real_pred).to(real_pred.device) * 0.9
        real_loss = F.binary_cross_entropy_with_logits(real_pred, target_real)
        
        # Fake loss
        target_fake = torch.zeros_like(fake_pred).to(fake_pred.device)
        fake_loss = F.binary_cross_entropy_with_logits(fake_pred, target_fake)
        
        return (real_loss + fake_loss) / 2