"""Evaluate a trained checkpoint on the TuSimple test split (IoU / Dice).

    .venv/bin/python evaluate.py --ckpt vgg_unet_lanenet.pt --max-samples 200
"""
import argparse
import random

import torch
from torch.utils.data import DataLoader

from common import get_device, eval_metrics
from dataset import TuSimpleLanes, load_records
from model import VGGUNet


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate VGG-UNet on TuSimple test set")
    p.add_argument("--data-root", default="TUSimple")
    p.add_argument("--test-dir", default="TUSimple/test_set")
    p.add_argument("--test-label", default="test_label.json")
    p.add_argument("--img-h", type=int, default=256)
    p.add_argument("--img-w", type=int, default=512)
    p.add_argument("--lane-width", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-samples", type=int, default=200, help="0 = use all test records")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--ckpt", default="vgg_unet_lanenet.pt")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()

    recs = load_records(args.data_root, [args.test_label])
    random.Random(42).shuffle(recs)
    if args.max_samples:
        recs = recs[:args.max_samples]

    ds = TuSimpleLanes(args.test_dir, recs, args.img_h, args.img_w, args.lane_width, train=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = VGGUNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    iou, dice = eval_metrics(model, dl, device)
    print(f"TEST  IoU {iou:.4f} | Dice {dice:.4f}  (n={len(ds)})")


if __name__ == "__main__":
    main()
