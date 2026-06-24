"""Save input / ground-truth / prediction overlay panels as PNGs (headless, no GUI needed).

    .venv/bin/python visualize.py --ckpt vgg_unet_lanenet.pt --n 4 --out preds.png
"""
import argparse
import random

import matplotlib
matplotlib.use("Agg")  # headless: render to file, no display
import matplotlib.pyplot as plt
import torch

from common import get_device, denorm
from dataset import TuSimpleLanes, load_records
from model import VGGUNet


def parse_args():
    p = argparse.ArgumentParser(description="Visualize VGG-UNet predictions to a PNG")
    p.add_argument("--data-root", default="TUSimple")
    p.add_argument("--split-dir", default="TUSimple/test_set",
                   help="folder that raw_file paths are relative to")
    p.add_argument("--label", default="test_label.json")
    p.add_argument("--img-h", type=int, default=256)
    p.add_argument("--img-w", type=int, default=512)
    p.add_argument("--lane-width", type=int, default=8)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--ckpt", default="vgg_unet_lanenet.pt")
    p.add_argument("--out", default="predictions.png")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = get_device()

    recs = load_records(args.data_root, [args.label])
    ds = TuSimpleLanes(args.split_dir, recs, args.img_h, args.img_w, args.lane_width, train=False)

    model = VGGUNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    idxs = random.Random(args.seed).sample(range(len(ds)), args.n)
    fig, ax = plt.subplots(args.n, 3, figsize=(15, 4 * args.n))
    if args.n == 1:
        ax = ax[None, :]
    for row, i in enumerate(idxs):
        x, y = ds[i]
        pred = torch.sigmoid(model(x.unsqueeze(0).to(device)))[0, 0].cpu().numpy()
        img = denorm(x)
        ax[row, 0].imshow(img); ax[row, 0].set_title("input"); ax[row, 0].axis("off")
        ax[row, 1].imshow(img); ax[row, 1].imshow(y[0], alpha=0.5, cmap="autumn")
        ax[row, 1].set_title("ground truth"); ax[row, 1].axis("off")
        ax[row, 2].imshow(img); ax[row, 2].imshow(pred > args.thr, alpha=0.5, cmap="autumn")
        ax[row, 2].set_title("prediction"); ax[row, 2].axis("off")
    plt.tight_layout()
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
