import torch
from torch.utils.data import Dataset, DataLoader, random_split 
from pathlib import Path
import rasterio
import numpy as np
import random

class BigEarthNetSRDataset(Dataset):
    def __init__(self, root_dir, subset_size=None, augment=True):
        self.root_dir = Path(root_dir) / 'BigEarthNet-S2' / 'BigEarthNet-S2'
        self.augment = augment
        
        # Band groupings by resolution
        self.bands_20m = ['B05', 'B06', 'B07', 'B11', 'B12', 'B8A']
        self.bands_60m = ['B01', 'B09']
        self.bands_10m = ['B02', 'B03', 'B04', 'B08']
        
        # Normalization parameters
        self.norm_params = {
            '10m': {'mean': [341.58, 638.64, 477.62, 3610.56],
                   'std': [196.76, 262.62, 343.52, 958.73]},
            '20m': {'mean': [1042.65, 2849.99, 3454.13, 1951.60, 1070.14, 3691.64],
                   'std': [363.53, 662.72, 854.12, 484.09, 435.56, 884.44]},
            '60m': {'mean': [314.72, 3692.39],
                   'std': [127.19, 726.79]}
        }
        
        # Initialize patches list
        print(f"Scanning directory: {self.root_dir}")
        self.patches = []
        
        for patch_dir in self.root_dir.iterdir():
            if patch_dir.is_dir():
                subdirs = [d for d in patch_dir.iterdir() if d.is_dir()]
                if subdirs:
                    self.patches.append((subdirs[0], subdirs[0].name))
                    
                    if subset_size and len(self.patches) >= subset_size:
                        break
        
        print(f"\nFound {len(self.patches)} patches")

    def normalize_bands(self, bands, resolution):
        """Improved normalization to prevent clipping"""
        bands = np.array(bands)
        mean = np.array(self.norm_params[resolution]['mean'])
        std = np.array(self.norm_params[resolution]['std'])
        
        # Normalize to zero mean and unit variance
        normalized = (bands - mean[:, None, None]) / (std[:, None, None] + 1e-8)
        
        # Clip outliers
        normalized = np.clip(normalized, -3, 3)
        
        # Scale to [-1, 1] range
        normalized = normalized / 3.0
        
        return normalized

    def apply_augmentation(self, bands_dict):
        """Apply data augmentation to all bands consistently"""
        if not self.augment:
            return bands_dict
            
        # Random horizontal flip
        if random.random() > 0.5:
            for key in bands_dict:
                bands_dict[key] = torch.flip(bands_dict[key], [-1])
                
        # Random vertical flip
        if random.random() > 0.5:
            for key in bands_dict:
                bands_dict[key] = torch.flip(bands_dict[key], [-2])
                
        return bands_dict

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch_dir, prefix = self.patches[idx]
        
        try:
            data = {}
            
            # Load and process each band group
            for band_group, band_list in [
                ('20m', self.bands_20m),
                ('60m', self.bands_60m),
                ('10m', self.bands_10m)
            ]:
                bands = []
                for band in band_list:
                    band_files = list(patch_dir.glob(f"*_{band}.tif"))
                    if not band_files:
                        print(f"Missing {band} in {patch_dir}")
                        return None
                        
                    with rasterio.open(band_files[0]) as src:
                        bands.append(src.read(1))
                
                # Stack and normalize
                bands = np.stack(bands)
                bands = self.normalize_bands(bands, band_group)
                data[f'bands_{band_group}'] = torch.from_numpy(bands).float()
            
            # Apply augmentation
            data = self.apply_augmentation(data)
            
            return data
            
        except Exception as e:
            print(f"Error loading patch {prefix}: {str(e)}")
            return None

def create_data_splits(dataset, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Split dataset into train, validation, and test sets"""
    total_size = len(dataset)
    
    # Calculate split sizes
    train_size = max(int(train_ratio * total_size), 2)
    val_size = max(int(val_ratio * total_size), 2)
    test_size = max(total_size - train_size - val_size, 2)
    
    # Create splits
    train_set, val_set, test_set = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    return train_set, val_set, test_set

def collate_fn(batch):
    """Custom collate function to handle None values"""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    
    return {
        key: torch.stack([item[key] for item in batch])
        for key in batch[0].keys()
    }

def create_dataloaders(train_set, val_set, test_set, batch_size=8, num_workers=0):
    """Create DataLoader objects for each dataset split"""
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader