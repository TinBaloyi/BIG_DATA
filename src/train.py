# src/train.py
"""
Training entrypoint.
Supports:
 - persistent cache via PersistentDataset (cache_dir on Drive)
 - resume auto from out_dir/checkpoints/last.ckpt
 - mixed precision (AMP)
 - grad accumulation (accum_steps)
 - sliding-window validation
"""

import os, argparse, time
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
import monai.transforms as mt

from dataset import build_file_list_from_roots, get_transforms, make_datasets
from model import build_unet3d
from utils import save_checkpoint, find_latest_checkpoint, load_checkpoint, set_seed
from monai.data import decollate_batch

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_roots", nargs="+", required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cache_rate", type=float, default=0.3)
    ap.add_argument("--use_persistent_cache", type=int, default=0)
    ap.add_argument("--cache_dir", type=str, default=None)
    ap.add_argument("--roi", type=int, nargs=3, default=[128,128,128])
    ap.add_argument("--pixdim", type=float, nargs=3, default=[1.0,1.0,1.0])
    ap.add_argument("--resume", choices=["auto","none","path"], default="auto")
    ap.add_argument("--resume_path", type=str, default="")
    ap.add_argument("--save_every", type=int, default=1)
    ap.add_argument("--keep_best", action="store_true")
    ap.add_argument("--accum_steps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    train_items, val_items = build_file_list_from_roots(args.data_roots, split_ratio=0.9, seed=args.seed)
    preproc, train_rand, val_t = get_transforms(tuple(args.pixdim), tuple(args.roi))
    train_ds, val_ds, collate_fn = make_datasets(train_items, val_items, preproc, train_rand,
                                                 use_persistent_cache=bool(args.use_persistent_cache),
                                                 cache_dir=args.cache_dir, cache_rate=args.cache_rate, num_workers=args.workers)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                              pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet3d(in_channels=4, num_classes=5).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler(enabled=args.amp)

    # resume logic
    start_epoch = 1
    best_metric = 0.0
    if args.resume == "auto":
        ck = find_latest_checkpoint(args.out_dir)
        if ck:
            st = load_checkpoint(ck, model, optimizer, scaler)
            if st:
                start_epoch = int(st.get("epoch", 0)) + 1
                best_metric = float(st.get("best_metric", 0.0) or 0.0)
                print(f"Resumed from {ck}, starting epoch {start_epoch}, best_metric={best_metric:.4f}")
    elif args.resume == "path" and args.resume_path:
        st = load_checkpoint(args.resume_path, model, optimizer, scaler)
        start_epoch = int(st.get("epoch", 0)) + 1

    loss_fn = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True, lambda_dice=1.0, lambda_ce=1.0)
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")

    for epoch in range(start_epoch, args.epochs+1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        step = 0
        for batch in train_loader:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)  # shape (B,1,D,H,W)
            with autocast(enabled=args.amp):
                logits = model(imgs)  # (B,5,D,H,W)
                loss = loss_fn(logits, labels)
                loss = loss / args.accum_steps
            scaler.scale(loss).backward()
            step += 1
            if step % args.accum_steps == 0:
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
            epoch_loss += float(loss.item() * args.accum_steps)

        avg_loss = epoch_loss / max(1, len(train_loader))
        print(f"Epoch {epoch}/{args.epochs} - train loss: {avg_loss:.4f}")

        # validation: sliding window inference
        model.eval()
        dice_vals = []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                labels = batch["label"].to(device)
                logits = sliding_window_inference(imgs, roi_size=tuple(args.roi), sw_batch_size=1, predictor=model, overlap=0.5)
                # argmax for voxel-wise
                probs = torch.softmax(logits, dim=1)
                pred_lbl = torch.argmax(probs, dim=1, keepdim=True)  # (B,1,D,H,W)
                # compute one-hot for classes 1..4
                onehot_pred = torch.nn.functional.one_hot(pred_lbl.squeeze(1).long(), num_classes=5).permute(0,4,1,2,3).float()
                onehot_gt = torch.nn.functional.one_hot(labels.squeeze(1).long(), num_classes=5).permute(0,4,1,2,3).float()
                # exclude background
                dice_metric(y_pred=onehot_pred[:,1:], y=onehot_gt[:,1:])
                dv = dice_metric.aggregate().item()
                dice_vals.append(dv)
                dice_metric.reset()

        mean_dice = float(sum(dice_vals) / max(1, len(dice_vals)))
        print(f"Epoch {epoch} - val voxel-wise mean dice (classes 1..4): {mean_dice:.4f}")

        # checkpointing
        if epoch % args.save_every == 0:
            save_checkpoint(args.out_dir, epoch, model, optimizer, scaler, best_metric, tag="last")
        if args.keep_best and mean_dice > best_metric:
            best_metric = mean_dice
            save_checkpoint(args.out_dir, epoch, model, optimizer, scaler, best_metric, tag="best")
            print(f"  ✓ new best {best_metric:.4f} saved.")

    # final save
    save_checkpoint(args.out_dir, args.epochs, model, optimizer, scaler, best_metric, tag="last")
    print("Training complete.")
if __name__ == "__main__":
    main()
