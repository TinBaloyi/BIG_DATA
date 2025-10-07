# src/infer.py
"""
Sliding-window inference. Produces one NIfTI per case: <case>_pred.nii.gz (labels 0..4).
"""

import os, argparse, glob
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, EnsureTyped
)
from monai.data import Dataset, DataLoader
from monai.transforms import AsDiscrete
from model import build_unet3d
from utils import find_latest_checkpoint, load_checkpoint

def build_test_items(roots):
    if isinstance(roots, str): roots=[roots]
    items=[]
    for r in roots:
        case_dirs = sorted([p for p in glob.glob(os.path.join(r, "*")) if os.path.isdir(p)])
        for d in case_dirs:
            case = os.path.basename(d.rstrip("/\\"))
            # find modality files (assumes same naming patterns used by dataset._find_file_for_case)
            def _find_variant(suffixes):
                for s in suffixes:
                    p = os.path.join(d, case + s)
                    if os.path.exists(p):
                        return p
                return None
            # common suffixes prioritized
            t1 = _find_variant(["-t1.nii.gz","-t1n.nii.gz","_t1.nii.gz","_t1n.nii.gz","-T1.nii.gz"])
            t1c = _find_variant(["-t1c.nii.gz","-t1ce.nii.gz","_t1c.nii.gz","_t1ce.nii.gz","-T1Gd.nii.gz"])
            t2 = _find_variant(["-t2w.nii.gz","-t2.nii.gz","_t2w.nii.gz","_t2.nii.gz","-T2.nii.gz"])
            flair = _find_variant(["-t2f.nii.gz","-flair.nii.gz","_t2f.nii.gz","_flair.nii.gz","-FLAIR.nii.gz"])
            if all([t1,t1c,t2,flair]):
                items.append({"image":[t1,t1c,t2,flair], "case":case})
            else:
                print(f"⚠️ missing modalities for inference case: {d}")
    return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_roots", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ckpt", default="auto")  # auto -> find latest in out_dir/checkpoints/last.ckpt
    ap.add_argument("--roi", type=int, nargs=3, default=[128,128,128])
    ap.add_argument("--pixdim", type=float, nargs=3, default=[1.0,1.0,1.0])
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.ckpt == "auto":
        # find latest among first out_dir (if provided)
        # pick the latest ckpt under parent out_dir of out_dir/preds if user passed same out_dir pattern
        # fallback to None
        ck = None
    else:
        ck = args.ckpt

    test_items = build_test_items(args.data_roots)
    pre = Compose = None
    pre = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=tuple(args.pixdim), mode=("bilinear",)*4),
        ScaleIntensityRanged(keys=["image"], a_min=[0,0,0,0], a_max=[3000]*4, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image"], source_key="image"),
        EnsureTyped(keys=["image"]),
    ]
    from monai.transforms import Compose
    pre = Compose(pre)
    ds = Dataset(test_items, pre)
    dl = DataLoader(ds, batch_size=1, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet3d(in_channels=4, num_classes=5).to(device)

    # try to auto-locate checkpoint if user passed out_dir style path
    if args.ckpt == "auto":
        # try to find latest under parent run directory of out_dir param (heuristic)
        # user can instead pass explicit --ckpt /path/to/checkpoint
        ck = None
        # scan common run paths mounted in workspace/runs/
        possible = []
        for p in glob.glob("/workspace/runs/*/checkpoints/last.ckpt") + glob.glob("/content/drive/MyDrive/*/checkpoints/last.ckpt"):
            possible.append(p)
        ck = possible[-1] if possible else None

    if ck is None:
        raise RuntimeError("No checkpoint provided/found. Use --ckpt /path/to/ckpt or place last.ckpt in known run folders.")
    state = load_checkpoint(ck, model, optimizer=None, scaler=None)
    print(f"Loaded checkpoint: {ck}")

    post = AsDiscrete(argmax=True, to_onehot= False)
    os.makedirs(args.out_dir, exist_ok=True)
    from monai.transforms import SaveImaged
    saver = SaveImaged(output_dir=args.out_dir, output_postfix="pred", output_ext=".nii.gz", separate_folder=False, resample=False)

    for b in dl:
        case = b["case"][0]
        img = b["image"].to(device)
        with torch.cuda.amp.autocast(enabled=args.amp):
            logits = sliding_window_inference(img, roi_size=tuple(args.roi), sw_batch_size=1, predictor=model, overlap=0.5)
            pred_lbl = torch.argmax(torch.softmax(logits, dim=1), dim=1, keepdim=True)  # (1,1,D,H,W)
        # convert to numpy-compatible and save
        b["pred"] = pred_lbl.cpu()
        saver(b, meta_keys="image_meta_dict", data_keys="pred")
        out_path = os.path.join(args.out_dir, f"{case}_pred.nii.gz")
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
