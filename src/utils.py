# src/utils.py
import os, glob, shutil, random
import torch
import numpy as np

def save_checkpoint(out_dir, epoch, model, optimizer=None, scaler=None, best_metric=None, tag="last"):
    ckts_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckts_dir, exist_ok=True)
    ckpt_name = f"{tag}_e{epoch:04d}.ckpt"
    ckpt_path = os.path.join(ckts_dir, ckpt_name)
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "scaler_state": scaler.state_dict() if scaler else None,
        "best_metric": best_metric
    }
    torch.save(payload, ckpt_path)
    # update symlink-friendly copies
    last_link = os.path.join(ckts_dir, "last.ckpt")
    best_link = os.path.join(ckts_dir, "best.ckpt")
    try:
        # copy to last.ckpt (overwrite)
        shutil.copyfile(ckpt_path, last_link)
        if tag == "best":
            shutil.copyfile(ckpt_path, best_link)
    except Exception:
        # fallback: write copies
        shutil.copyfile(ckpt_path, last_link)
    return ckpt_path

def find_latest_checkpoint(out_dir):
    ckts_dir = os.path.join(out_dir, "checkpoints")
    if not os.path.exists(ckts_dir): return None
    # prefer last.ckpt copy
    last = os.path.join(ckts_dir, "last.ckpt")
    if os.path.exists(last): return last
    # otherwise find newest .ckpt
    ckpts = sorted(glob.glob(os.path.join(ckts_dir, "*.ckpt")))
    return ckpts[-1] if ckpts else None

def load_checkpoint(path, model, optimizer=None, scaler=None, map_location=None):
    if path is None or not os.path.exists(path):
        return None
    if map_location is None:
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model_state"])
    if optimizer and state.get("optimizer_state"):
        optimizer.load_state_dict(state["optimizer_state"])
    if scaler and state.get("scaler_state"):
        scaler.load_state_dict(state["scaler_state"])
    return state

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
