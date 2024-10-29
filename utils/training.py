import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from .metrics import EvaluationMetrics
from losses.losses import SRGANLoss

class SRGANTrainer:
    def __init__(self, generator, discriminator, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing trainer on device: {self.device}")
        
        # Models
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        
        # Initialize criterion
        self.criterion = SRGANLoss(
            device=self.device,
            lambda_content=config.lambda_content,
            lambda_perceptual=config.lambda_perceptual,
            lambda_adversarial=config.lambda_adversarial,
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
         
        # Initialize schedulers
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
        # Initialize training state
        self.current_epoch = 0
        self.best_psnr = -float('inf')
        self.train_history = {
            'g_losses': [],
            'd_losses': [],
            'psnr': [],
            'ssim': [],
            'fid': []
        }
        # Metrics and logging
        self.metrics = EvaluationMetrics(self.device)
        self._setup_directories()

    def _setup_directories(self):
        """Create necessary directories for saving outputs"""
        # Main directories
        self.save_dir = Path(self.config.save_dir)
        self.log_dir = Path(self.config.log_dir)
        
        # Create samples directory as a subdirectory of save_dir
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.samples_dir = self.save_dir / 'samples' / timestamp
        
        # Create all directories
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Created output directories:")
        print(f"- Checkpoints: {self.save_dir}")
        print(f"- Logs: {self.log_dir}")
        print(f"- Samples: {self.samples_dir}")

    def train(self, train_loader, val_loader, num_epochs,callback=None):
        print(f"Starting training on {self.device}")
        print(f"Training set size: {len(train_loader.dataset)}")
        print(f"Validation set size: {len(val_loader.dataset)}")
        
        best_psnr = 0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Training phase
            train_metrics = self._train_epoch(train_loader, epoch)
            
            # Validation phase
            val_metrics = self._validate(val_loader)
            
            # Update history
            self.train_history['g_losses'].append(train_metrics['generator_losses'])
            self.train_history['d_losses'].append(train_metrics['discriminator_losses'])
            self.train_history['psnr'].append(val_metrics['PSNR'])
            self.train_history['ssim'].append(val_metrics['SSIM'])
            self.train_history['fid'].append(val_metrics['FID'])

            # Step schedulers
            self.g_scheduler.step()
            self.d_scheduler.step()
            
            # Save best model
            if val_metrics['PSNR'] > best_psnr:
                best_psnr = val_metrics['PSNR']
                self._save_checkpoint(epoch, val_metrics)
            
            # Generate and save samples
            if (epoch + 1) % self.config.save_interval == 0:
                self._generate_samples(val_loader, f"epoch_{epoch+1}")
            
            # Call callback if provided
            if callback is not None:
                callback(epoch, train_metrics, val_metrics)
            # Log progress
            self._print_epoch_summary(epoch, train_metrics, val_metrics, time.time() - start_time)

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

    def _train_epoch(self, train_loader, epoch):
        self.generator.train()
        self.discriminator.train()
        self.current_epoch = epoch

        # Learning rate warmup
        # Step schedulers if we're past warmup
        if epoch >= self.config.warmup_epochs:
            self.g_scheduler.step()
            self.d_scheduler.step()
        # Initialize metrics
        train_metrics = {
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
            train_metrics['generator_losses'].append(g_loss.item())
            train_metrics['discriminator_losses'].append(d_loss.item())
            for key, value in components.items():
                train_metrics['component_losses'][key].append(value)

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
               for k, v in train_metrics.items()}

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

        # Add noise for stability
        noise_factor = 0.05
        real_noisy = real_hr + noise_factor * torch.randn_like(real_hr)
        fake_noisy = fake_hr + noise_factor * torch.randn_like(fake_hr)

        # Real loss with label smoothing
        real_pred = self.discriminator(real_noisy)
        real_label = torch.ones_like(real_pred).to(self.device) * 0.9

        # Fake loss
        fake_pred = self.discriminator(fake_noisy.detach())
        fake_label = torch.zeros_like(fake_pred).to(self.device)

        # Use smoother loss function
        d_loss_real = F.binary_cross_entropy_with_logits(real_pred, real_label)
        d_loss_fake = F.binary_cross_entropy_with_logits(fake_pred, fake_label)
        d_loss = (d_loss_real + d_loss_fake) / 2

        # Clip loss value
        d_loss = torch.clamp(d_loss, 0.0, 2.0)

        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 
                                     self.config.gradient_clip)
        self.d_optimizer.step()

        return d_loss
    def _train_generator(self, real_hr, x_20m, x_60m):
        self.g_optimizer.zero_grad()
        
        # Generate fake images
        fake_hr = self.generator(x_20m, x_60m)
        
        # Get discriminator predictions
        fake_pred = self.discriminator(fake_hr)
        
        # Calculate losses
        g_loss, components = self.criterion.generator_loss(fake_hr, real_hr, fake_pred)
        
        # Apply epoch-based scaling
        epoch_factor = min(1.0, self.current_epoch / 50)
        scaled_loss = g_loss * (1.0 + epoch_factor * 0.5)  # Gradual increase
        
        # Backward and optimize
        scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.generator.parameters(), 
            self.config.gradient_clip
        )
        self.g_optimizer.step()
        
        return scaled_loss, components

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
                
                # Get all metrics
                batch_metrics = self.metrics.evaluate_batch(fake_hr, real_hr)
                val_metrics.append(batch_metrics)
                
                # Optional: Save sample images periodically
                if len(val_metrics) == 1:  # First batch
                    self.metrics.visualize_batch_comparison(
                        fake_hr, real_hr,
                        save_path=f"{self.save_dir}/validation_sample_epoch_{self.current_epoch}.png"
                    )
        
        # Average metrics
        avg_metrics = {}
        for metric in val_metrics[0].keys():
            avg_metrics[metric] = np.mean([m[metric] for m in val_metrics])
        
        # Print metrics
        self.metrics.print_metrics(avg_metrics, prefix="Validation ")
        
        return avg_metrics

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """
        Compute gradient penalty for WGAN-GP
        Args:
            real_samples: Real high-resolution images
            fake_samples: Generated high-resolution images
        """
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

    def _generate_samples(self, val_loader, epoch_name):
        """Generate and save sample SR images"""
        self.generator.eval()
        with torch.no_grad():
            try:
                batch = next(iter(val_loader))
                if batch is not None:
                    x_20m = batch['bands_20m'].to(self.device)
                    x_60m = batch['bands_60m'].to(self.device)
                    real_hr = batch['bands_10m'].to(self.device)
                    
                    # Generate SR image
                    fake_hr = self.generator(x_20m, x_60m)
                    
                    # Function to prepare images for display
                    def prepare_for_display(tensor):
                        # Take first 3 channels for RGB display
                        tensor = tensor[:, :3]
                        # Convert from [-1, 1] to [0, 1]
                        tensor = (tensor + 1) / 2
                        tensor = torch.clamp(tensor, 0, 1)
                        return tensor.cpu().numpy().transpose(0, 2, 3, 1)

                    # Prepare images
                    sr_img = prepare_for_display(fake_hr)[0]
                    hr_img = prepare_for_display(real_hr)[0]
                    lr_img = prepare_for_display(x_20m)[0]  # Using 20m as LR reference
                    
                    # Create figure
                    plt.figure(figsize=(15, 5))
                    
                    # Plot LR input
                    plt.subplot(131)
                    plt.imshow(lr_img)
                    plt.title('Low Resolution Input')
                    plt.axis('off')
                    
                    # Plot SR output
                    plt.subplot(132)
                    plt.imshow(sr_img)
                    plt.title('Super Resolution')
                    plt.axis('off')
                    
                    # Plot HR ground truth
                    plt.subplot(133)
                    plt.imshow(hr_img)
                    plt.title('High Resolution (Ground Truth)')
                    plt.axis('off')
                    
                    # Save with proper path handling
                    save_path = self.samples_dir / f"sample_{epoch_name}.png"
                    plt.savefig(save_path, bbox_inches='tight', dpi=300)
                    plt.close()
                    
                    print(f"Saved sample images to {save_path}")
            
            except Exception as e:
                print(f"Error generating samples: {str(e)}")
                traceback.print_exc()

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

    def _print_epoch_summary(self, epoch, train_metrics, val_metrics, elapsed_time):
        """Print a summary of the epoch's metrics"""
        print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
        print(f"Time elapsed: {elapsed_time:.2f}s")
        
        print("\nTraining Metrics:")
        if isinstance(train_metrics, dict):
            if 'generator_losses' in train_metrics:
                print(f"Generator Loss: {train_metrics['generator_losses']:.4f}")
            if 'discriminator_losses' in train_metrics:
                print(f"Discriminator Loss: {train_metrics['discriminator_losses']:.4f}")
            if 'component_losses' in train_metrics:
                print("\nComponent Losses:")
                for component, value in train_metrics['component_losses'].items():
                    if isinstance(value, (float, int)):
                        print(f"{component}: {value:.4f}")
                    elif isinstance(value, list) and value:
                        print(f"{component}: {np.mean(value):.4f}")
        
        print("\nValidation Metrics:")
        for key, value in val_metrics.items():
            print(f"{key}: {value:.4f}")

        # Print current learning rates
        print("\nLearning Rates:")
        print(f"Generator: {self.g_optimizer.param_groups[0]['lr']:.2e}")
        print(f"Discriminator: {self.d_optimizer.param_groups[0]['lr']:.2e}")

    def _save_checkpoint(self, epoch, metrics):
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'g_scheduler_state_dict': self.g_scheduler.state_dict(),
            'd_scheduler_state_dict': self.d_scheduler.state_dict(),
            'metrics': metrics,
            'train_history': self.train_history,
            'best_psnr': self.best_psnr,
            'current_epoch': self.current_epoch
        }
        
        torch.save(checkpoint, f"{self.config.save_dir}/best_model.pth")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        self.g_scheduler.load_state_dict(checkpoint['g_scheduler_state_dict'])
        self.d_scheduler.load_state_dict(checkpoint['d_scheduler_state_dict'])
        self.current_epoch = checkpoint['current_epoch']
        self.best_psnr = checkpoint['best_psnr']
        self.train_history = checkpoint['train_history']
        
        return checkpoint['epoch'], checkpoint['metrics']