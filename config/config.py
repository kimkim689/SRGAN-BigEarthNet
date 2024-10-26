from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class TrainingConfig:
    # Generator Configuration
    n_res_blocks: int = 16
    base_channels: int = 64
    growth_rate: int = 32
    attention_reduction: int = 16
    use_dense_blocks: bool = True
    use_attention: bool = True
    
    # Discriminator Configuration
    disc_channels: int = 64
    use_spectral_norm: bool = True
    num_discriminators: int = 2        # Number of discriminator scales
    patch_size: int = 16              # PatchGAN size
         

    # Training Configuration
    batch_size: int = 4
    num_epochs: int = 100
    lr_generator: float = 1e-5
    lr_discriminator: float = 5e-5  # TTUR: 4x generator lr
    beta1: float = 0.5
    beta2: float = 0.999
    lr_decay_epochs: int = 10
    lr_decay_factor: float = 0.5
    d_train_freq: int = 2
    
    # Loss Weights
    lambda_content: float = 5.0
    lambda_perceptual: float = 0.5
    lambda_adv: float = 0.005
    lambda_spectral: float = 0.2
    
    # Training Optimizations
    use_mixed_precision: bool = True
    num_workers: int = 4
    accumulation_steps: int = 8
    ema_decay: float = 0.999
    
    # Data Augmentation
    aug_flip_prob: float = 0.5
    aug_rotate_prob: float = 0.3
    
    # Stability
    label_smoothing: float = 0.2
    noise_std: float = 0.01
    gradient_clip: float = 0.03
    clip_value: float = 0.005
    n_critic: int = 5
    warmup_epochs: int = 5
    
    # Directory settings
    save_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Training settings
    save_interval: int = 5    
    log_interval: int = 10     

    def __post_init__(self):
        """Called after dataclass initialization"""
        self.create_directories()
    
    def create_directories(self):
        """Create necessary directories for saving outputs"""
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

# Usage
config = TrainingConfig()