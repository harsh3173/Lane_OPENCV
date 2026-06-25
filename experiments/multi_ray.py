"""Multi-source ray-cast — cast fans from several bottom-row origins to SEGREGATE drivable segments
(lane 1, lane 2, ...). Standalone experiment; does NOT modify ray_mask.py / ray_pilot.py.

Idea: a single bottom-centre source answers "where can I drive". Several sources offset across the
bottom (left-mid / centre / right-mid) each cast their own fan; the UNION's connected components are
the distinct drivable segments. Cast with `--no-yellow-pass` so the yellow centre line is a BOUNDARY
→ left/right lanes fall into separate components → coloured as separate lanes.

    .venv/bin/python experiments/multi_ray.py --img-dir tub_generated_track --n 6 --out multi_preview.png
    .venv/bin/python experiments/multi_ray.py --img-dir tub_old_car/images --video --vid-out multi_oldcar.mp4
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root -> import raypilot

import cv2
import numpy as np

from raypilot.ray_mask import calibrate, seed_ref, list_imgs, numeric_key   # read-only reuse

SEG_COLORS = [(0, 255, 0), (255, 80, 80), (0, 180, 255), (255, 0, 255), (0, 255, 255), (160, 120, 255)]


def cast_fan(lab, hsv, ref, ref_v, sx, sy, n_rays, a0, a1, white_margin, white_s,
             color_thr, wl, horizon, edge_thr, edge_window, yellow_pass, consec=3, step=1.0):
    """One fan from origin (sx, sy) -> endpoint list (same march rules as ray_mask.cast_rays)."""
    H, W = lab.shape[:2]
    wvec = np.array([wl, 1.0, 1.0], np.float32)
    y_top = horizon * H
    eps = []
    for ang in np.linspace(a0, a1, n_rays):
        rad = np.deg2rad(ang); dx, dy = np.cos(rad), -np.sin(rad)
        lastx, lasty, bad, t, trail = sx, sy, 0, step, []
        while True:
            x, y = sx + dx * t, sy + dy * t
            xi, yi = int(round(x)), int(round(y))
            if xi < 0 or xi >= W or yi < 0 or yi >= H or y < y_top:
                break
            h, s, v = hsv[yi, xi, 0], hsv[yi, xi, 1], hsv[yi, xi, 2]
            labp = lab[yi, xi]
            is_yellow = yellow_pass and (15 <= h <= 40) and s > 80 and v > 80
            is_white = v > ref_v + white_margin and s < white_s
            cdiff = np.sqrt(np.sum(wvec * (labp - ref) ** 2))
            edge = len(trail) >= edge_window and np.linalg.norm(labp - trail[-edge_window]) > edge_thr
            if not is_yellow and (is_white or edge or cdiff > color_thr):
                bad += 1
                if bad >= consec:
                    break
            else:
                bad = 0; lastx, lasty = x, y
            trail.append(labp); t += step
        eps.append((lastx, lasty))
    return eps


def fan_mask(shape, seed, eps):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.array([seed] + eps, np.int32)], 255)
    return m


def segment_frame(bgr, ref, ref_v, sources, kw, min_seg_frac=0.01):
    """Return (segment_label_map, n_segments, per_source_seeds). Segments = connected components of
    the union of all source fans (each a distinct drivable corridor)."""
    H, W = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    union = np.zeros((H, W), np.uint8)
    seeds, source_masks = [], []
    for fx in sources:
        sx, sy = fx * W, kw["seed_y"] * H
        seeds.append((sx, sy))
        eps = cast_fan(lab, hsv, ref, ref_v, sx, sy, kw["n_rays"], kw["a0"], kw["a1"],
                       kw["white_margin"], kw["white_s"], kw["color_thr"], kw["wl"], kw["horizon"],
                       kw["edge_thr"], kw["edge_window"], kw["yellow_pass"])
        m = fan_mask((H, W), (sx, sy), eps)
        source_masks.append(m)
        union = cv2.bitwise_or(union, m)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(union, 8)
    sid = sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_seg_frac * H * W)
    return source_masks, sid, seeds


def overlay(bgr, source_masks, nseg, seeds, scale=3):
    """Colour each SOURCE's fan distinctly (overlaps blend) so the multi-source coverage is visible."""
    H, W = bgr.shape[:2]; s = scale
    canvas = cv2.resize(bgr, (W * s, H * s), interpolation=cv2.INTER_LINEAR)
    for i, m in enumerate(source_masks):
        mm = cv2.resize(m, (W * s, H * s), interpolation=cv2.INTER_NEAREST)
        tint = canvas.copy(); tint[mm > 0] = SEG_COLORS[i % len(SEG_COLORS)]
        canvas = cv2.addWeighted(canvas, 0.72, tint, 0.28, 0)
    for i, (sx, sy) in enumerate(seeds):
        cv2.circle(canvas, (int(sx * s), int(sy * s)), 6, SEG_COLORS[i % len(SEG_COLORS)], -1)
        cv2.circle(canvas, (int(sx * s), int(sy * s)), 6, (255, 255, 255), 1)
    cv2.putText(canvas, f"{len(source_masks)} sources | {nseg} segment(s)", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return canvas


def parse_args():
    p = argparse.ArgumentParser(description="Multi-source ray-cast lane/segment segregation")
    p.add_argument("--img-dir", required=True)
    p.add_argument("--sources", type=float, nargs="+", default=[0.25, 0.5, 0.75],
                   help="seed x-fractions across the bottom row (same level, offset)")
    p.add_argument("--seed-y", type=float, default=0.85)
    p.add_argument("--n-rays", type=int, default=60)
    p.add_argument("--a0", type=float, default=8)
    p.add_argument("--a1", type=float, default=172)
    p.add_argument("--white-margin", type=int, default=90)
    p.add_argument("--white-s", type=int, default=60)
    p.add_argument("--color-thr", type=float, default=40)
    p.add_argument("--wl", type=float, default=0.1)
    p.add_argument("--horizon", type=float, default=0.35)
    p.add_argument("--edge-thr", type=float, default=22)
    p.add_argument("--edge-window", type=int, default=4)
    p.add_argument("--no-yellow-pass", dest="yellow_pass", action="store_false", default=False,
                   help="(default) yellow line is a BOUNDARY -> lanes segregate")
    p.add_argument("--yellow-pass", dest="yellow_pass", action="store_true")
    p.add_argument("--calib-sample", type=int, default=150)
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="multi_preview.png")
    p.add_argument("--video", action="store_true")
    p.add_argument("--vid-out", default="multi_overlay.mp4")
    p.add_argument("--max-frames", type=int, default=1200)
    p.add_argument("--fps", type=int, default=20)
    return p.parse_args()


def kwdict(a):
    return dict(seed_y=a.seed_y, n_rays=a.n_rays, a0=a.a0, a1=a.a1, white_margin=a.white_margin,
               white_s=a.white_s, color_thr=a.color_thr, wl=a.wl, horizon=a.horizon,
               edge_thr=a.edge_thr, edge_window=a.edge_window, yellow_pass=a.yellow_pass)


def main():
    a = parse_args()
    paths = sorted(list_imgs(a.img_dir), key=numeric_key)
    if not paths:
        print("no images"); return
    ref, ref_v = calibrate(paths, a.calib_sample)
    print(f"global ref LAB({ref[0]:.0f},{ref[1]:.0f},{ref[2]:.0f}) V{ref_v:.0f} | sources {a.sources} "
          f"| yellow_pass={a.yellow_pass}")
    kw = kwdict(a)

    if a.video:
        seq = paths[:a.max_frames] if a.max_frames else paths
        H, W = cv2.imread(seq[0]).shape[:2]
        writer = cv2.VideoWriter(a.vid_out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W * 3, H * 3))
        seg_counts = []
        for p in seq:
            bgr = cv2.imread(p)
            source_masks, nseg, seeds = segment_frame(bgr, ref, ref_v, a.sources, kw)
            seg_counts.append(nseg)
            writer.write(overlay(bgr, source_masks, nseg, seeds))
        writer.release()
        print(f"wrote {a.vid_out} ({len(seq)} frames) | avg segments {np.mean(seg_counts):.1f}")
        return

    import random
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sel = [paths[i] for i in random.Random(a.seed).sample(range(len(paths)), min(a.n, len(paths)))]
    fig, ax = plt.subplots(len(sel), 2, figsize=(7, 3 * len(sel)))
    if len(sel) == 1:
        ax = ax[None, :]
    for r, p in enumerate(sel):
        bgr = cv2.imread(p); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        source_masks, nseg, seeds = segment_frame(bgr, ref, ref_v, a.sources, kw)
        ov = cv2.cvtColor(overlay(bgr, source_masks, nseg, seeds, scale=1), cv2.COLOR_BGR2RGB)
        ax[r, 0].imshow(rgb); ax[r, 0].axis("off"); ax[r, 0].set_title("input")
        ax[r, 1].imshow(ov); ax[r, 1].axis("off"); ax[r, 1].set_title(f"{nseg} segment(s)")
    plt.tight_layout(); fig.savefig(a.out, dpi=110, bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
