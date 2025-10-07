# src/dataset.py
"""
Filelist builder, transforms, and dataset helpers.
- Supports flexible modality suffixes (t1n/t1/t1ce/t1c/t2w/t2/t2f/flair/...).
- Deterministic heavy preprocess (for caching) + light random augmentations per epoch.
- Offers PersistentDataset (disk cache) and CacheDataset (in-RAM) options.
"""

import os
import glob
from typing import List, Tuple, Dict, Optional
import torch
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, RandSpatialCropd,
    RandFlipd, RandShiftIntensityd, RandGaussianNoised,
    EnsureTyped, Compose
)
from monai.data import PersistentDataset, CacheDataset, Dataset
from monai.data import list_data_collate

# Common modality suffix variants we detect (ordered preferred)
MODALITY_CANDIDATES = {
    "t1": ["-t1.nii.gz", "-t1n.nii.gz", "_t1.nii.gz", "_t1n.nii.gz", "-T1.nii.gz"],
    "t1ce": ["-t1c.nii.gz", "-t1ce.nii.gz", "_t1c.nii.gz", "_t1ce.nii.gz", "-T1Gd.nii.gz", "_T1Gd.nii.gz"],
    "t2": ["-t2w.nii.gz", "-t2.nii.gz", "_t2w.nii.gz", "_t2.nii.gz", "-T2.nii.gz"],
    "flair": ["-t2f.nii.gz", "-flair.nii.gz", "_t2f.nii.gz", "_flair.nii.gz", "-FLAIR.nii.gz"],
    "seg": ["-seg.nii.gz", "_seg.nii.gz", "-segmentation.nii.gz", "_label.nii.gz"]
}

def _find_file_for_case(case_dir: str, modality_keys=MODALITY_CANDIDATES) -> Optional[Dict]:
    case = os.path.basename(case_dir.rstrip("/\\"))
    files = {}
    for key, patt_list in modality_keys.items():
        found = None
        for patt in patt_list:
            p = os.path.join(case_dir, case + patt)
            if os.path.exists(p):
                found = p
                break
        files[key] = found
    # require images and seg for training items
    if all(files.get(k) for k in ("t1","t1ce","t2","flair")):
        return {
            "image": [files["t1"], files["t1ce"], files["t2"], files["flair"]],
            "label": files.get("seg"),  # may be None for test-only cases
            "case": case
        }
    return None

def build_file_list_from_roots(roots: List[str], split_ratio: float = 0.9, seed: int = 42) -> Tuple[List, List]:
    """
    roots: list of directories that each contain many case-folders
    returns: train_items, val_items  (each is a list of dicts accepted by MONAI Dataset)
    """
    if isinstance(roots, str):
        roots = [roots]
    case_dirs = []
    for r in roots:
        case_dirs += [p for p in sorted(glob.glob(os.path.join(r, "*"))) if os.path.isdir(p)]
    items = []
    for d in case_dirs:
        it = _find_file_for_case(d)
        if it:
            if it["label"] is None:
                # test-only case (skip for train/val)
                continue
            items.append(it)
        else:
            print(f"⚠️ missing modalities for case dir: {d}")
    if len(items) == 0:
        raise RuntimeError("No valid cases found. Check dataset root paths and filename suffixes.")
    # deterministic split
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(items), generator=g).tolist()
    ntrain = int(split_ratio * len(items))
    train_items = [items[i] for i in idx[:ntrain]]
    val_items = [items[i] for i in idx[ntrain:]]
    return train_items, val_items

def get_transforms(pixdim=(1.0,1.0,1.0), roi=(128,128,128)):
    """
    Returns (preproc_transform, train_rand_transform, val_transform)
    - preproc_transform is deterministic and good to cache.
    - train_rand_transform applied on top of cached outputs during training (keeps randomness).
    """
    preproc = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),   # image: (4,D,H,W), label: (1,D,H,W)
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"],
                             a_min=[0,0,0,0], a_max=[3000,3000,3000,3000],
                             b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image","label"], source_key="image"),
        EnsureTyped(keys=["image", "label"])
    ])
    train_rand = Compose([
        RandSpatialCropd(keys=["image","label"], roi_size=roi, random_center=True, random_size=False),
        RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=[0]),
        RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=[1]),
        RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=[2]),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),
        RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.01),
    ])
    val = Compose([
        # same as preproc + optional center-crop
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"],
                             a_min=[0,0,0,0], a_max=[3000,3000,3000,3000],
                             b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image","label"], source_key="image"),
        EnsureTyped(keys=["image","label"])
    ])
    return preproc, train_rand, val

def make_datasets(train_items, val_items, preproc_transform, train_rand_transform,
                  use_persistent_cache: bool=False, cache_dir: str=None, cache_rate: float=0.3,
                  num_workers: int = 4):
    """
    Returns (train_ds, val_ds) where train_ds is a Dataset that applies random augs on top of cached preproc outputs.
    If use_persistent_cache True: uses PersistentDataset (disk-backed) with preproc_transform (cache_dir required).
    Otherwise: uses CacheDataset or direct Dataset depending on memory.
    """
    if use_persistent_cache:
        if cache_dir is None:
            raise ValueError("cache_dir must be provided for persistent cache.")
        os.makedirs(cache_dir, exist_ok=True)
        pre_train_ds = PersistentDataset(data=train_items, transform=preproc_transform, cache_dir=cache_dir)
        pre_val_ds = PersistentDataset(data=val_items, transform=preproc_transform, cache_dir=cache_dir)
    else:
        # CacheDataset keeps cached items in RAM (partially) based on cache_rate
        pre_train_ds = CacheDataset(data=train_items, transform=preproc_transform, cache_rate=cache_rate, num_workers=num_workers)
        pre_val_ds = CacheDataset(data=val_items, transform=preproc_transform, cache_rate=cache_rate, num_workers=num_workers)

    train_ds = Dataset(pre_train_ds, transform=train_rand_transform)
    val_ds = Dataset(pre_val_ds, transform=None)
    return train_ds, val_ds, list_data_collate
