import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from datetime import datetime
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from .metrics import EvaluationMetrics
from losses.losses import SRGANLoss
import os

class SRGANTrainer:
    def __init__(self, generator, discriminator, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing trainer on device: {self.device}")
        
        # Models
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.current_step = 0

        # Initialize criterion
        self.criterion = SRGANLoss(
            device=self.device,
            lambda_content=config.lambda_content,
            lambda_perceptual=config.lambda_perceptual,
            lambda_adv=config.lambda_adv,
            lambda_spectral=config.lambda_spectral,
            label_smoothing=config.label_smoothing
        )
        
        # Optimizers
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=config.lr_generator,
            betas=(config.beta1, config.beta2)
        )
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config.lr_discriminator,
            betas=(config.beta1, config.beta2)
        )
        
        # Schedulers
        self.g_scheduler = torch.optim.lr_scheduler.StepLR(
            self.g_optimizer,
            step_size=config.lr_decay_epochs,
            gamma=config.lr_decay_factor
        )
        self.d_scheduler = torch.optim.lr_scheduler.StepLR(
            self.d_optimizer,
            step_size=config.lr_decay_epochs,
            gamma=config.lr_decay_factor
        )
        
        # Metrics and logging
        self.metrics = EvaluationMetrics(self.device)
        
        # Training state
        self.current_epoch = 0
        self.best_psnr = float('-inf')
        
        # Create directories
        self.save_dir = Path(config.save_dir)
        self.log_dir = Path(config.log_dir)
        self.samples_dir = self.save_dir / "samples"

        # Create all directories
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Compute gradient penalty for WGAN-GP"""
        # Random weight term for interpolation
        alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(self.device)
        
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + 
                       (1 - alpha) * fake_samples).requires_grad_(True)
        
        # Get discriminator output for interpolated images
        d_interpolates = self.discriminator(interpolates)
        
        # Get gradients with respect to inputs
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates).to(self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Compute gradient penalty
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty

    def train(self, train_loader, val_loader, num_epochs,callback=None):
        print(f"Starting training on {self.device}")
        print(f"Training set size: {len(train_loader.dataset)}")
        print(f"Validation set size: {len(val_loader.dataset)}")
        
        # Initialize tracking
        training_history = {
            'generator_losses': [],
            'discriminator_losses': [],
            'validation_metrics': {'PSNR': [], 'SSIM': [], 'FID': [], 'MSE':[]},
            'learning_rates': {'generator': [], 'discriminator': []}
        }
        
        best_psnr = 0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self._train_epoch(train_loader, epoch)
            
            # Validation phase
            val_metrics = self._validate(val_loader)
            
            # Update history
            self._update_history(training_history, train_metrics, val_metrics)
            
            # Save best model
            if val_metrics['PSNR'] > self.best_psnr:
                self.best_psnr = val_metrics['PSNR']
                self._save_checkpoint(epoch, val_metrics)
            
            # Generate samples periodically
            if (epoch + 1) % self.config.save_interval == 0:
                self._generate_samples(val_loader, f"epoch_{epoch+1}")
        
            # Update learning rates
            self.g_scheduler.step()
            self.d_scheduler.step()

            # Call callback if provided
            if callback is not None:
                callback(epoch, train_metrics, val_metrics)

            # Print progress
            self._print_epoch_summary(epoch, train_metrics, val_metrics, time.time() - start_time)

            # Save history
            self._save_history(training_history)

    def _train_epoch(self, train_loader, epoch):
        self.generator.train()
        self.discriminator.train()
        self.current_epoch = epoch

        # Learning rate warmup
        if epoch < self.config.warmup_epochs:
            lr_scale = min(1., (epoch + 1) / self.config.warmup_epochs)
            for param_group in self.g_optimizer.param_groups:
                param_group['lr'] = self.config.lr_generator * lr_scale
            for param_group in self.d_optimizer.param_groups:
                param_group['lr'] = self.config.lr_discriminator * lr_scale

        # Initialize metrics
        epoch_metrics = {
            'generator_losses': [],
            'discriminator_losses': [],
            'component_losses': {
                'content': [], 
                'perceptual': [], 
                'adversarial': [], 
                'total': []
            }
        }

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config.num_epochs}')
        for batch_idx, batch in enumerate(pbar):
            self.current_step = epoch * len(train_loader) + batch_idx
            if batch is None:
                continue
            
            # Get data
            x_20m = batch['bands_20m'].to(self.device)
            x_60m = batch['bands_60m'].to(self.device)
            real_hr = batch['bands_10m'].to(self.device)

            # Train discriminator with frequency control
            if batch_idx % self.config.d_train_freq == 0:
                d_loss = self._train_discriminator(real_hr, x_20m, x_60m)
            else:
                d_loss = torch.tensor(0.0).to(self.device)

            # Train generator
            g_loss, components = self._train_generator(real_hr, x_20m, x_60m)

            # Monitor loss ratio
            g_d_ratio = g_loss.item() / (d_loss.item() + 1e-8)
            if g_d_ratio > 5.0 or g_d_ratio < 0.2:
                print(f"\nWarning: G/D loss ratio: {g_d_ratio:.2f}")

            # Update metrics
            epoch_metrics['generator_losses'].append(g_loss.item())
            epoch_metrics['discriminator_losses'].append(d_loss.item())
            for key in epoch_metrics['component_losses'].keys():  # Only track initialized components
                if key in components:
                    epoch_metrics['component_losses'][key].append(components[key])
            # Update progress bar
            pbar.set_postfix({
                'G_loss': f"{g_loss.item():.4f}",
                'D_loss': f"{d_loss.item():.4f}",
                'Content': f"{components['content']:.4f}"
            })

        # Log gradient norms at the end of epoch
        g_norm, d_norm = self._log_gradients()
        print(f"Gradient norms - G: {g_norm:.4f}, D: {d_norm:.4f}")

        # Calculate epoch averages
        return {k: np.mean(v) if isinstance(v, list) else 
               {k2: np.mean(v2) for k2, v2 in v.items()}
               for k, v in epoch_metrics.items()}

    def _log_gradients(self):
        """Helper method to compute gradient norms"""
        total_norm_g = 0
        total_norm_d = 0
        for p in self.generator.parameters():
            if p.grad is not None:
                total_norm_g += p.grad.norm(2).item() ** 2
        for p in self.discriminator.parameters():
            if p.grad is not None:
                total_norm_d += p.grad.norm(2).item() ** 2
        return np.sqrt(total_norm_g), np.sqrt(total_norm_d)

    def _train_discriminator(self, real_hr, x_20m, x_60m):
        self.d_optimizer.zero_grad()

        # Generate fake samples
        with torch.no_grad():
            fake_hr = self.generator(x_20m, x_60m)

        # Add noise for robustness
        noise_factor = 0.05
        real_noisy = real_hr + noise_factor * torch.randn_like(real_hr).to(self.device)
        fake_noisy = fake_hr.detach() + noise_factor * torch.randn_like(fake_hr).to(self.device)

        # Get predictions
        real_outputs = self.discriminator(real_noisy)
        fake_outputs = self.discriminator(fake_noisy)

        # Calculate loss
        d_loss = self.criterion.discriminator_loss(real_outputs, fake_outputs)

        # Backward and optimize
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.discriminator.parameters(),
            self.config.gradient_clip
        )
        self.d_optimizer.step()

        return d_loss

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

    def _train_generator(self, real_hr, x_20m, x_60m):
        # Generate fake images
        fake_hr = self.generator(x_20m, x_60m)
        fake_pred = self.discriminator(fake_hr)

        # Calculate losses
        g_loss, components = self.criterion.generator_loss(fake_hr, real_hr, fake_pred)

        # Scale loss for gradient accumulation
        g_loss = g_loss / self.config.accumulation_steps

        # Backward pass
        g_loss.backward()

        # Update weights if accumulation steps reached
        if (self.current_step + 1) % self.config.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(), 
                self.config.gradient_clip
            )
            self.g_optimizer.step()
            self.g_optimizer.zero_grad()

        return g_loss * self.config.accumulation_steps, components
        
        

    def _validate(self, val_loader):
        self.generator.eval()
        val_metrics = []
        
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                
                x_20m = batch['bands_20m'].to(self.device)
                x_60m = batch['bands_60m'].to(self.device)
                real_hr = batch['bands_10m'].to(self.device)
                
                fake_hr = self.generator(x_20m, x_60m)
                batch_metrics = self.metrics.evaluate_batch(fake_hr, real_hr)
                val_metrics.append(batch_metrics)
        
        # Average metrics
        avg_metrics = {}
        for metric in val_metrics[0].keys():
            avg_metrics[metric] = np.mean([m[metric] for m in val_metrics])
        
        return avg_metrics

    def _generate_samples(self, val_loader, epoch_name):
        """Generate and save sample images"""
        self.generator.eval()
        with torch.no_grad():
            batch = next(iter(val_loader))
            if batch is not None:
                x_20m = batch['bands_20m'].to(self.device)
                x_60m = batch['bands_60m'].to(self.device)
                real_hr = batch['bands_10m'].to(self.device)
                
                fake_hr = self.generator(x_20m, x_60m)
                
                # Proper normalization for visualization
                def normalize_for_display(tensor):
                    # Convert from [-1, 1] to [0, 1]
                    tensor = (tensor + 1) / 2
                    tensor = torch.clamp(tensor, 0, 1)
                    return tensor.cpu().numpy().transpose(1, 2, 0)
                
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                plt.imshow(normalize_for_display(x_20m[0, :3]))
                plt.title('Input (20m bands)')
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.imshow(normalize_for_display(fake_hr[0, :3]))
                plt.title('Generated SR')
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.imshow(normalize_for_display(real_hr[0, :3]))
                plt.title('Real HR')
                plt.axis('off')
                
                # Save with proper path handling
                save_path = self.samples_dir / f"sample_{epoch_name}.png"
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                plt.close()

    def _save_checkpoint(self, epoch, metrics):
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'g_scheduler_state_dict': self.g_scheduler.state_dict(),
            'd_scheduler_state_dict': self.d_scheduler.state_dict(),
            'metrics': metrics
        }
        
        torch.save(checkpoint, self.save_dir / "best_model.pth")

    def _save_history(self, history):
        with open(self.log_dir / "training_history.json", 'w') as f:
            json.dump(history, f, indent=4)

    def _update_history(self, history, train_metrics, val_metrics):
        history['generator_losses'].append(train_metrics['generator_losses'])
        history['discriminator_losses'].append(train_metrics['discriminator_losses'])
        
        for metric, value in val_metrics.items():
            history['validation_metrics'][metric].append(value)
        
        history['learning_rates']['generator'].append(
            self.g_optimizer.param_groups[0]['lr']
        )
        history['learning_rates']['discriminator'].append(
            self.d_optimizer.param_groups[0]['lr']
        )
    def visualize_batch_statistics(batch):
        """monitor data"""
        for key in batch:
            tensor = batch[key]
            print(f"\n{key}:")
            print(f"Range: [{tensor.min():.3f}, {tensor.max():.3f}]")
            print(f"Mean: {tensor.mean():.3f}")
            print(f"Std: {tensor.std():.3f}")

    def add_gradient_monitoring(model):
        """Add to your training loop"""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        print(f"Gradient norm: {total_norm:.3f}")

    def _print_epoch_summary(self, epoch, train_metrics, val_metrics, elapsed_time):
        print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
        print(f"Time elapsed: {elapsed_time:.2f}s")
        print("\nTraining Metrics:")
        print(f"Generator Loss: {train_metrics['generator_losses']:.4f}")
        print(f"Discriminator Loss: {train_metrics['discriminator_losses']:.4f}")
        print("\nValidation Metrics:")
        for key, value in val_metrics.items():
            print(f"{key}: {value:.4f}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        self.g_scheduler.load_state_dict(checkpoint['g_scheduler_state_dict'])
        self.d_scheduler.load_state_dict(checkpoint['d_scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['metrics']
    
    def monitor_training(self, epoch, g_loss, d_loss, metrics):
        """Add this to your trainer"""
        # Monitor loss ratio
        g_d_ratio = g_loss / (d_loss + 1e-8)

        # Check for potential problems
        if g_d_ratio > 5.0:
            print("Warning: Generator loss much higher than discriminator")
            # Consider reducing generator learning rate

        if g_d_ratio < 0.1:
            print("Warning: Discriminator loss much higher than generator")
            # Consider reducing discriminator learning rate

        # Monitor gradients
        def get_gradient_norm(model):
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.norm(2).item() ** 2
            return total_norm ** 0.5

        g_grad_norm = get_gradient_norm(self.generator)
        d_grad_norm = get_gradient_norm(self.discriminator)

        print(f"Gradient norms - G: {g_grad_norm:.4f}, D: {d_grad_norm:.4f}")