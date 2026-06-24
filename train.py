"""Train the VGG-UNet lane detector on TuSimple.

Quick MVP run:   .venv/bin/python train.py --epochs 1 --max-samples 100
Full run:        .venv/bin/python train.py --epochs 15 --max-samples 0
"""
import argparse
import os

import torch
from tqdm import tqdm

from common import get_device, make_criterion, eval_metrics
from dataset import build_train_val_loaders
from model import VGGUNet


def parse_args():
    p = argparse.ArgumentParser(description="Train VGG-UNet lane detector")
    p.add_argument("--train-dir", default="TUSimple/train_set")
    p.add_argument("--img-h", type=int, default=256)
    p.add_argument("--img-w", type=int, default=512)
    p.add_argument("--lane-width", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--max-samples", type=int, default=800,
                   help="cap training records for speed; 0 = use all")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--ckpt", default="vgg_unet_lanenet.pt")
    p.add_argument("--no-pretrained", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")

    train_ds, val_ds, train_dl, val_dl = build_train_val_loaders(
        args.train_dir, args.img_h, args.img_w, args.lane_width, args.batch_size,
        val_frac=args.val_frac, max_samples=args.max_samples or None,
        num_workers=args.num_workers,
    )
    print(f"train {len(train_ds)} | val {len(val_ds)}")

    model = VGGUNet(pretrained=not args.no_pretrained).to(device)
    criterion = make_criterion(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_dice = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_dl, desc=f"epoch {epoch}/{args.epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running / max(len(train_dl), 1)
        if device.type == "mps":
            torch.mps.empty_cache()
        val_iou, val_dice = eval_metrics(model, val_dl, device)
        print(f"epoch {epoch}: loss {train_loss:.4f} | val IoU {val_iou:.4f} | val Dice {val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), args.ckpt)
            print(f"  saved {args.ckpt} (Dice {best_dice:.4f})")

    print(f"done. best val Dice {best_dice:.4f} -> {os.path.abspath(args.ckpt)}")


if __name__ == "__main__":
    main()
