"""TuSimple lane-segmentation dataset.

Each label line gives a clip's last frame: `lanes` (x per height) paired with shared `h_samples`
(y). `x == -2` means no lane at that height. Lane polylines are rasterized into a binary mask.
`raw_file` paths are relative to the split folder (train_set / test_set), not the repo root.
"""
import json
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from common import IMAGENET_MEAN, IMAGENET_STD

TRAIN_LABELS = ["label_data_0313.json", "label_data_0531.json", "label_data_0601.json"]


def load_records(label_dir: str, label_files):
    recs = []
    for lf in label_files:
        with open(os.path.join(label_dir, lf)) as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    return recs


def lane_mask(lanes, h_samples, h, w, thickness):
    """Rasterize lane polylines into a binary (h, w) uint8 mask."""
    mask = np.zeros((h, w), np.uint8)
    for lane in lanes:
        pts = [(int(x), int(y)) for x, y in zip(lane, h_samples) if x >= 0]
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            cv2.line(mask, (x0, y0), (x1, y1), 1, thickness)
    return mask


class TuSimpleLanes(Dataset):
    def __init__(self, root, records, img_h, img_w, lane_width=8, train=True):
        self.root = root
        self.records = records
        self.img_h, self.img_w = img_h, img_w
        self.lane_width = lane_width
        self.train = train

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        img = cv2.imread(os.path.join(self.root, r["raw_file"]))
        if img is None:
            raise FileNotFoundError(os.path.join(self.root, r["raw_file"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]
        mask = lane_mask(r["lanes"], r["h_samples"], h0, w0, self.lane_width)

        img = cv2.resize(img, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)

        if self.train and random.random() < 0.5:  # horizontal flip
            img, mask = img[:, ::-1], mask[:, ::-1]

        img = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        mask = torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0)
        return img, mask


def build_train_val_loaders(train_dir, img_h, img_w, lane_width, batch_size,
                            val_frac=0.1, max_samples=None, num_workers=2, seed=42):
    records = load_records(train_dir, TRAIN_LABELS)
    random.Random(seed).shuffle(records)
    if max_samples:
        records = records[:max_samples]
    n_val = int(len(records) * val_frac)
    val_recs, train_recs = records[:n_val], records[n_val:]

    train_ds = TuSimpleLanes(train_dir, train_recs, img_h, img_w, lane_width, train=True)
    val_ds = TuSimpleLanes(train_dir, val_recs, img_h, img_w, lane_width, train=False)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_ds, val_ds, train_dl, val_dl
