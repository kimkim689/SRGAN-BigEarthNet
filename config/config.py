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
    
    # Training Configuration
    batch_size: int = 16
    num_epochs: int = 100
    lr_generator: float = 5e-4
    lr_discriminator: float = 1e-4  # TTUR: 4x generator lr
    beta1: float = 0.5
    beta2: float = 0.999
    lr_decay_epochs: int = 10
    lr_decay_factor: float = 0.5
    
    # Loss Weights
    lambda_content: float = 1.0
    lambda_perceptual: float = 0.2
    lambda_adversarial: float = 0.001
    lambda_spectral: float = 0.1
    
    # Training Optimizations
    use_mixed_precision: bool = True
    num_workers: int = 4
    accumulation_steps: int = 4
    ema_decay: float = 0.999
    warmup_epochs: int = 5
    d_train_freq: int = 2  

    # Data Augmentation
    aug_flip_prob: float = 0.5
    aug_rotate_prob: float = 0.3
    
    # Stability
    label_smoothing: float = 0.15
    noise_std: float = 0.01
    gradient_clip: float = 0.1
    
    # Directory settings
    save_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Training settings
    save_interval: int = 5     # Changed to type annotation
    log_interval: int = 10      # Changed to type annotation
    
    def __post_init__(self):
        """Called after dataclass initialization"""
        self.create_directories()
    
    def create_directories(self):
        """Create necessary directories for saving outputs"""
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
