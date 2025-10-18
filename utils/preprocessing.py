"""
BraTS Adult Glioma Data Preprocessing
Handles NIfTI file loading, normalization, and preparation
"""

import os
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Tuple, Dict, List
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split # type: ignore
import SimpleITK as sitk


class BraTSPreprocessor:
    """Preprocessor for BraTS Adult Glioma dataset"""
    
    def __init__(self, raw_data_dir: str, processed_data_dir: str):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Modality suffixes
        self.modalities = {
            't1n': 'T1',      # Native T1
            't1c': 'T1Gd',    # Post-contrast T1
            't2w': 'T2',      # T2-weighted
            't2f': 'FLAIR'    # FLAIR
        }
        
        # Label mapping
        self.label_map = {
            0: 'Background',
            1: 'NETC',  # Non-enhancing tumor core
            2: 'SNFH',  # Surrounding non-enhancing FLAIR hyperintensity
            3: 'ET',    # Enhancing tumor
            4: 'RC'     # Resection cavity
        }
        
    def find_patient_folders(self) -> List[Path]:
        """Find all patient folders in raw data directory"""
        patient_folders = sorted([
            d for d in self.raw_data_dir.iterdir() 
            if d.is_dir() and d.name.startswith('BraTS-GLI')
        ])
        print(f"Found {len(patient_folders)} patient folders")
        return patient_folders
    
    def load_nifti(self, filepath: Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
        """Load NIfTI file and return data + header"""
        nifti_img = nib.load(str(filepath))
        data = nifti_img.get_fdata()
        return data, nifti_img
    
    def normalize_intensity(self, volume: np.ndarray, 
                           modality: str) -> np.ndarray:
        """
        Normalize MRI intensity values
        Different normalization for different modalities
        """
        # Create mask of non-zero values (brain region)
        mask = volume > 0
        
        if not mask.any():
            return volume
        
        # Z-score normalization within brain region
        brain_voxels = volume[mask]
        mean = np.mean(brain_voxels)
        std = np.std(brain_voxels)
        
        if std > 0:
            volume_norm = volume.copy()
            volume_norm[mask] = (volume[mask] - mean) / std
            return volume_norm
        else:
            return volume
    
    def load_patient_data(self, patient_folder: Path) -> Dict[str, np.ndarray]:
        """Load all modalities and segmentation for a patient"""
        patient_id = patient_folder.name
        data = {}
        
        # Load all modalities
        for suffix, modality in self.modalities.items():
            filepath = patient_folder / f"{patient_id}-{suffix}.nii.gz"
            if filepath.exists():
                volume, _ = self.load_nifti(filepath)
                # Normalize intensity
                volume_norm = self.normalize_intensity(volume, modality)
                data[modality] = volume_norm
            else:
                print(f"Warning: Missing {modality} for {patient_id}")
                return None
        
        # Load segmentation (ground truth)
        seg_filepath = patient_folder / f"{patient_id}-seg.nii.gz"
        if seg_filepath.exists():
            segmentation, _ = self.load_nifti(seg_filepath)
            data['segmentation'] = segmentation.astype(np.uint8)
        else:
            print(f"Warning: Missing segmentation for {patient_id}")
            data['segmentation'] = None  # For test set
        
        return data
    
    def create_composite_labels(self, segmentation: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Create composite label masks for evaluation
        - Tumor Core (TC): ET + NETC (labels 3 + 1)
        - Whole Tumor (WT): ET + SNFH + NETC (labels 3 + 2 + 1)
        """
        composite = {}
        
        # Individual labels
        composite['ET'] = (segmentation == 3).astype(np.uint8)
        composite['NETC'] = (segmentation == 1).astype(np.uint8)
        composite['SNFH'] = (segmentation == 2).astype(np.uint8)
        composite['RC'] = (segmentation == 4).astype(np.uint8)
        
        # Composite labels
        composite['TC'] = ((segmentation == 3) | (segmentation == 1)).astype(np.uint8)
        composite['WT'] = ((segmentation == 3) | (segmentation == 2) | 
                          (segmentation == 1)).astype(np.uint8)
        
        return composite
    
    def crop_to_nonzero(self, volumes: Dict[str, np.ndarray], 
                        margin: int = 5) -> Tuple[Dict[str, np.ndarray], Tuple]:
        """
        Crop volumes to non-zero region (brain) with margin
        Returns cropped volumes and bounding box coordinates
        """
        # Create combined mask from all modalities
        combined_mask = np.zeros_like(list(volumes.values())[0], dtype=bool)
        for key, vol in volumes.items():
            if key != 'segmentation':
                combined_mask |= (vol > 0)
        
        # Find bounding box
        coords = np.argwhere(combined_mask)
        if len(coords) == 0:
            return volumes, None
        
        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0)
        
        # Add margin
        shape = combined_mask.shape
        z_min = max(0, z_min - margin)
        y_min = max(0, y_min - margin)
        x_min = max(0, x_min - margin)
        z_max = min(shape[0], z_max + margin)
        y_max = min(shape[1], y_max + margin)
        x_max = min(shape[2], x_max + margin)
        
        bbox = (z_min, z_max, y_min, y_max, x_min, x_max)
        
        # Crop all volumes
        cropped = {}
        for key, vol in volumes.items():
            cropped[key] = vol[z_min:z_max, y_min:y_max, x_min:x_max]
        
        return cropped, bbox
    
    def preprocess_dataset(self, crop: bool = True, 
                          test_size: float = 0.15,
                          val_size: float = 0.15):
        """
        Preprocess entire dataset
        - Load all patients
        - Normalize intensities
        - Optionally crop to brain region
        - Split into train/val/test
        """
        patient_folders = self.find_patient_folders()
        
        processed_data = []
        failed_cases = []
        
        print("Processing patients...")
        for patient_folder in tqdm(patient_folders):
            patient_id = patient_folder.name
            
            # Load patient data
            data = self.load_patient_data(patient_folder)
            if data is None:
                failed_cases.append(patient_id)
                continue
            
            # Crop to brain region
            if crop:
                data, bbox = self.crop_to_nonzero(data)
            else:
                bbox = None
            
            # Save processed data
            output_dir = self.processed_data_dir / patient_id
            output_dir.mkdir(exist_ok=True)
            
            # Save volumes as numpy arrays (faster loading)
            for key, volume in data.items():
                np.save(output_dir / f"{key}.npy", volume)
            
            # Save metadata
            metadata = {
                'patient_id': patient_id,
                'original_shape': [int(x) for x in volume.shape],
                'bbox': [int(x) for x in bbox] if bbox is not None else None,
                'modalities': list(self.modalities.values()),
                'has_segmentation': data['segmentation'] is not None
            }
            
            with open(output_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            processed_data.append(patient_id)
        
        print(f"\nSuccessfully processed: {len(processed_data)} patients")
        print(f"Failed cases: {len(failed_cases)}")
        
        # Create train/val/test splits
        self.create_splits(processed_data, test_size, val_size)
        
        return processed_data, failed_cases
    
    def create_splits(self, patient_ids: List[str], 
                     test_size: float, val_size: float):
        """Create and save train/val/test splits"""
        # First split: train+val vs test
        train_val_ids, test_ids = train_test_split(
            patient_ids, test_size=test_size, random_state=42
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        train_ids, val_ids = train_test_split(
            train_val_ids, test_size=val_size_adjusted, random_state=42
        )
        
        # Save splits
        splits_dir = self.processed_data_dir.parent / 'splits'
        splits_dir.mkdir(exist_ok=True)
        
        splits = {
            'train': train_ids,
            'val': val_ids,
            'test': test_ids
        }
        
        for split_name, ids in splits.items():
            with open(splits_dir / f'{split_name}.txt', 'w') as f:
                f.write('\n'.join(ids))
        
        print(f"\nDataset splits:")
        print(f"  Train: {len(train_ids)} patients")
        print(f"  Val:   {len(val_ids)} patients")
        print(f"  Test:  {len(test_ids)} patients")
        
        # Save split info as JSON
        split_info = {
            'total': len(patient_ids),
            'train': len(train_ids),
            'val': len(val_ids),
            'test': len(test_ids),
            'test_size': test_size,
            'val_size': val_size
        }
        
        with open(splits_dir / 'split_info.json', 'w') as f:
            json.dump(split_info, f, indent=2)


if __name__ == '__main__':
    # Configuration
    RAW_DATA_DIR = r'C:\Projects\BIG_DATA\BIG_DATA\data\raw\training_data1_v2'
    PROCESSED_DATA_DIR = r'C:\Projects\BIG_DATA\BIG_DATA\data\processed'  # Update this
    
    # Initialize preprocessor
    preprocessor = BraTSPreprocessor(RAW_DATA_DIR, PROCESSED_DATA_DIR)
    
    # Run preprocessing
    processed_patients, failed = preprocessor.preprocess_dataset(
        crop=True,
        test_size=0.15,
        val_size=0.15
    )
    
    print("\nPreprocessing complete!")
    