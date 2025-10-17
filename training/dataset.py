"""
PyTorch Dataset and DataLoader for BraTS Adult Glioma
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
from typing import Dict, Tuple, Optional, List
import albumentations as A
from scipy import ndimage


class BraTSDataset(Dataset):
    """PyTorch Dataset for BraTS Adult Glioma"""
    
    def __init__(self, 
                 processed_data_dir: str,
                 patient_ids: List[str],
                 patch_size: Tuple[int, int, int] = (128, 128, 128),
                 mode: str = 'train',
                 transforms: Optional[callable] = None):
        """
        Args:
            processed_data_dir: Path to processed data
            patient_ids: List of patient IDs to include
            patch_size: Size of patches to extract (D, H, W)
            mode: 'train', 'val', or 'test'
            transforms: Data augmentation transforms
        """
        self.processed_data_dir = Path(processed_data_dir)
        self.patient_ids = patient_ids
        self.patch_size = patch_size
        self.mode = mode
        self.transforms = transforms
        
        self.modalities = ['T1', 'T1Gd', 'T2', 'FLAIR']
        
    def __len__(self) -> int:
        return len(self.patient_ids)
    
    def load_patient(self, patient_id: str) -> Dict[str, np.ndarray]:
        """Load all modalities and segmentation for a patient"""
        patient_dir = self.processed_data_dir / patient_id
        
        data = {}
        
        # Load modalities
        for modality in self.modalities:
            filepath = patient_dir / f"{modality}.npy"
            data[modality] = np.load(filepath)
        
        # Load segmentation if exists
        seg_filepath = patient_dir / "segmentation.npy"
        if seg_filepath.exists():
            data['segmentation'] = np.load(seg_filepath)
        else:
            data['segmentation'] = None
        
        # Load metadata
        with open(patient_dir / 'metadata.json', 'r') as f:
            data['metadata'] = json.load(f)
        
        return data
    
    def extract_random_patch(self, 
                            volumes: Dict[str, np.ndarray], 
                            segmentation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract random patch from volumes
        Weighted sampling: higher probability for patches with tumor
        """
        D, H, W = volumes[self.modalities[0]].shape
        pd, ph, pw = self.patch_size
        
        # For training, sample patches with tumor more frequently
        if self.mode == 'train' and segmentation is not None:
            # Create weight map (higher weight where tumor exists)
            tumor_mask = segmentation > 0
            
            # Downsample tumor mask for efficiency
            scale = 4
            tumor_mask_ds = tumor_mask[::scale, ::scale, ::scale]
            
            # Calculate valid starting positions
            valid_d = max(1, D - pd)
            valid_h = max(1, H - ph)
            valid_w = max(1, W - pw)
            
            # Create probability map
            prob_map = np.zeros((valid_d // scale, valid_h // scale, valid_w // scale))
            for d in range(0, valid_d, scale):
                for h in range(0, valid_h, scale):
                    for w in range(0, valid_w, scale):
                        patch_tumor = tumor_mask_ds[
                            d//scale:min((d+pd)//scale, tumor_mask_ds.shape[0]),
                            h//scale:min((h+ph)//scale, tumor_mask_ds.shape[1]),
                            w//scale:min((w+pw)//scale, tumor_mask_ds.shape[2])
                        ]
                        prob_map[d//scale, h//scale, w//scale] = patch_tumor.sum()
            
            # Add base probability to avoid zero probability patches
            prob_map = prob_map + 0.1
            prob_map = prob_map.flatten()
            prob_map = prob_map / prob_map.sum()
            
            # Sample position
            idx = np.random.choice(len(prob_map), p=prob_map)
            d_idx = (idx // (prob_map.shape[0] * prob_map.shape[1])) * scale
            h_idx = ((idx % (prob_map.shape[0] * prob_map.shape[1])) // prob_map.shape[1]) * scale
            w_idx = (idx % prob_map.shape[1]) * scale
            
            d_start = min(d_idx, D - pd)
            h_start = min(h_idx, H - ph)
            w_start = min(w_idx, W - pw)
        else:
            # Random sampling for validation/test
            d_start = np.random.randint(0, max(1, D - pd + 1))
            h_start = np.random.randint(0, max(1, H - ph + 1))
            w_start = np.random.randint(0, max(1, W - pw + 1))
        
        # Extract patches
        image_patch = np.stack([
            volumes[mod][d_start:d_start+pd, 
                        h_start:h_start+ph, 
                        w_start:w_start+pw]
            for mod in self.modalities
        ], axis=0)
        
        if segmentation is not None:
            seg_patch = segmentation[d_start:d_start+pd,
                                    h_start:h_start+ph,
                                    w_start:w_start+pw]
        else:
            seg_patch = np.zeros((pd, ph, pw), dtype=np.uint8)
        
        return image_patch, seg_patch
    
    def apply_augmentation(self, 
                          image: np.ndarray, 
                          segmentation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation"""
        if self.transforms is None:
            return image, segmentation
        
        # Apply transforms
        augmented = self.transforms(image=image, mask=segmentation)
        return augmented['image'], augmented['mask']
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample"""
        patient_id = self.patient_ids[idx]
        
        # Load patient data
        patient_data = self.load_patient(patient_id)
        
        # Extract volumes
        volumes = {mod: patient_data[mod] for mod in self.modalities}
        segmentation = patient_data['segmentation']
        
        # Extract patch
        image_patch, seg_patch = self.extract_random_patch(volumes, segmentation)
        
        # Apply augmentation (only for training)
        if self.mode == 'train' and self.transforms is not None:
            image_patch, seg_patch = self.apply_augmentation(image_patch, seg_patch)
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image_patch).float()
        seg_tensor = torch.from_numpy(seg_patch).long()
        
        return {
            'image': image_tensor,
            'segmentation': seg_tensor,
            'patient_id': patient_id
        }


class BraTSAugmentation:
    """Data augmentation for 3D medical images"""
    
    @staticmethod
    def get_train_transforms():
        """Get training augmentation pipeline"""
        return A.Compose([
            # Random flip
            A.HorizontalFlip(p=0.5),
            # Random rotation (small angles for medical images)
            A.Rotate(limit=15, p=0.5),
            # Random brightness/contrast
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            # Random gamma
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            # Gaussian noise
            A.GaussNoise(var_limit=(0.0, 0.05), p=0.3),
        ])
    
    @staticmethod
    def get_val_transforms():
        """Get validation transforms (usually none)"""
        return None


def get_dataloaders(processed_data_dir: str,
                   splits_dir: str,
                   batch_size: int = 2,
                   num_workers: int = 4,
                   patch_size: Tuple[int, int, int] = (128, 128, 128)):
    """
    Create train/val/test dataloaders
    
    Args:
        processed_data_dir: Path to processed data
        splits_dir: Path to train/val/test split files
        batch_size: Batch size
        num_workers: Number of workers for dataloader
        patch_size: Patch size for training
    
    Returns:
        Dictionary of dataloaders
    """
    splits_path = Path(splits_dir)
    
    # Load patient IDs for each split
    def load_split(split_name):
        with open(splits_path / f'{split_name}.txt', 'r') as f:
            return [line.strip() for line in f.readlines()]
    
    train_ids = load_split('train')
    val_ids = load_split('val')
    test_ids = load_split('test')
    
    # Create datasets
    train_dataset = BraTSDataset(
        processed_data_dir=processed_data_dir,
        patient_ids=train_ids,
        patch_size=patch_size,
        mode='train',
        transforms=BraTSAugmentation.get_train_transforms()
    )
    
    val_dataset = BraTSDataset(
        processed_data_dir=processed_data_dir,
        patient_ids=val_ids,
        patch_size=patch_size,
        mode='val',
        transforms=None
    )
    
    test_dataset = BraTSDataset(
        processed_data_dir=processed_data_dir,
        patient_ids=test_ids,
        patch_size=patch_size,
        mode='test',
        transforms=None
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


if __name__ == '__main__':
    # Test dataset
    dataloaders = get_dataloaders(
        processed_data_dir='/path/to/processed/data',
        splits_dir='/path/to/splits',
        batch_size=2,
        num_workers=4
    )
    
    # Test loading a batch
    for batch in dataloaders['train']:
        print("Image shape:", batch['image'].shape)
        print("Segmentation shape:", batch['segmentation'].shape)
        print("Patient IDs:", batch['patient_id'])
        break