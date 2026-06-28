"""Radial ray-cast drivable mask (standalone; does not touch the other pipelines).

Like a 2D lidar from the car: fan rays out from the bottom-centre of the frame and march each one
upward/outward until it hits a STOP -- either a white lane line (bright + low-saturation) or too big
a colour jump from the track (LAB distance to the seed colour). The polygon swept by the ray
endpoints is the drivable region. Stops at lane lines (white) and at off-track edges (grass/floor),
so it works for the sim road and the old-car mat without per-domain polarity.

    .venv/bin/python ray_mask.py --img-dir tub_generated_track --n 6 --out ray_preview.png
    .venv/bin/python ray_mask.py --img-dir tub_old_car/images --n 6 --out ray_preview_oldcar.png
    .venv/bin/python ray_mask.py --img-dir DIR --mask-dir DIR/ray_masks   # write masks
    .venv/bin/python ray_mask.py --img-dir DIR --video --vid-out rays.mp4 # overlay video
"""
import argparse
import glob
import os

import cv2
import numpy as np
# matplotlib is imported lazily inside the preview path only (keeps the Raspberry-Pi runtime,
# which just needs cv2 + numpy, free of a matplotlib dependency).


SEED_BOX = (0.42, 0.58, 0.80, 0.88)


def seed_ref(bgr, seed_box=SEED_BOX):
    """Median (LAB, V) of the bottom-centre seed patch for one frame."""
    H, W = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    x0, x1, y0, y1 = int(seed_box[0] * W), int(seed_box[1] * W), int(seed_box[2] * H), int(seed_box[3] * H)
    return np.median(lab[y0:y1, x0:x1].reshape(-1, 3), axis=0), float(np.median(hsv[y0:y1, x0:x1, 2]))


def calibrate(paths, sample=150, seed_box=SEED_BOX):
    """One fixed track reference for the whole run: median across many frames' seed patches.

    Robust to the few off-track frames (median ignores them), and -- crucially -- it does NOT drift to
    the floor when off-track, so off-track frames diverge from it and the rays collapse.
    """
    sel = paths[:: max(1, len(paths) // sample)]
    refs, vs = zip(*(seed_ref(cv2.imread(p), seed_box) for p in sel))
    return np.median(np.array(refs), axis=0), float(np.median(vs))


def cast_rays(bgr, seed_y=0.85, seed_box=SEED_BOX, n_rays=80,
              a0=8, a1=172, white_margin=45, white_s=60, color_thr=40, consec=3, step=1.0, wl=0.15,
              ref=None, ref_v=None, horizon=0.0, edge_thr=22, edge_window=4, yellow_pass=True,
              shadow_pass=False, green_stop=True, green_s=20, green_dark=15):
    """Return (endpoints [(x,y)] in angle order, seed (x,y), ref_lab). Rays march until a STOP.

    A ray stops on:
      - white lane line: low saturation AND value > road_value + white_margin (RELATIVE brightness,
        so a uniformly-bright grey road is not mistaken for a white line), or
      - off-colour edge: weighted-LAB distance to the reference > color_thr. L is down-weighted (wl)
        so the road's own brightness gradient doesn't stop the ray; chroma (grass/floor) does.

    ref/ref_v: fixed run-level track colour (from calibrate). If None, fall back to this frame's seed.
    """
    H, W = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    wvec = np.array([wl, 1.0, 1.0], np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    if ref is None or ref_v is None:
        fr, fv = seed_ref(bgr, seed_box)
        ref = fr if ref is None else ref
        ref_v = fv if ref_v is None else ref_v
    sx, sy = W / 2.0, seed_y * H
    y_top = horizon * H                                # rays may not climb above the horizon (sky guard)
    eps = []
    for ang in np.linspace(a0, a1, n_rays):
        rad = np.deg2rad(ang)
        dx, dy = np.cos(rad), -np.sin(rad)            # image y points down -> negate for "up"
        lastx, lasty = sx, sy
        bad = 0
        t = step
        trail = []                                     # LAB history along this ray (for local edge)
        while True:
            x, y = sx + dx * t, sy + dy * t
            xi, yi = int(round(x)), int(round(y))
            if xi < 0 or xi >= W or yi < 0 or yi >= H or y < y_top:
                break
            h, s, v = hsv[yi, xi, 0], hsv[yi, xi, 1], hsv[yi, xi, 2]
            labp = lab[yi, xi]
            # a yellow lane line is DRIVABLE (the car may cross it) -> never a stop, so the ray
            # passes through it and coverage doesn't collapse when crossing/straddling the line
            is_yellow = yellow_pass and (15 <= h <= 40) and s > 80 and v > 80
            is_white = v > ref_v + white_margin and s < white_s
            # GRASS = off-track: green hue + darker than the road. Brightness-independent (works for
            # dark or bright green) and robust where grass differs from grey road mainly in L, not
            # chroma -> the weighted-LAB term alone can't catch it. Yellow line is excluded (it passes).
            is_grass = green_stop and (35 <= h <= 90) and s > green_s and v < ref_v - green_dark
            if shadow_pass:
                # SHADOW-ROBUST: a shadow preserves chroma but drops lightness, so a DARKER-but-neutral
                # pixel is road in shade (drivable). Penalise only POSITIVE dL (brighter -> desert /
                # pavement edge / glare); darkening is free. Off-road still trips via chroma (grass /
                # brown tree trunk) or the bright term. Local edge uses CHROMA only, so a shadow's hard
                # brightness boundary doesn't stop the ray, while a coloured edge (foliage/dirt) does.
                dL = max(float(labp[0] - ref[0]), 0.0)
                cdiff = np.sqrt(wl * dL * dL + (labp[1] - ref[1]) ** 2 + (labp[2] - ref[2]) ** 2)
                edge = (len(trail) >= edge_window and
                        np.linalg.norm(labp[1:] - trail[-edge_window][1:]) > edge_thr)
            else:
                cdiff = np.sqrt(np.sum(wvec * (labp - ref) ** 2))        # vs global ref -> off-track
                # local edge: sharp FULL-LAB jump vs a few px back -> a white line or grass boundary,
                # while the road's own smooth gradient stays below the threshold
                edge = len(trail) >= edge_window and np.linalg.norm(labp - trail[-edge_window]) > edge_thr
            if not is_yellow and (is_white or is_grass or edge or cdiff > color_thr):
                bad += 1
                if bad >= consec:
                    break
            else:
                bad = 0
                lastx, lasty = x, y                   # advance the confirmed free endpoint
            trail.append(labp)
            t += step
        eps.append((lastx, lasty))
    return eps, (sx, sy), ref


def ray_mask(bgr, **kw):
    """Drivable mask = filled polygon of seed + ray endpoints."""
    H, W = bgr.shape[:2]
    eps, seed, _ = cast_rays(bgr, **kw)
    poly = np.array([seed] + eps, np.int32)
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    return mask, eps, seed


def parse_args():
    p = argparse.ArgumentParser(description="Radial ray-cast drivable mask")
    p.add_argument("--img-dir", required=True)
    p.add_argument("--mask-dir", default=None, help="write masks here")
    p.add_argument("--n-rays", type=int, default=80)
    p.add_argument("--a0", type=float, default=8, help="start angle (deg; 90=straight up)")
    p.add_argument("--a1", type=float, default=172, help="end angle")
    p.add_argument("--seed-y", type=float, default=0.85, help="ray origin y (above the bottom hood/shadow)")
    p.add_argument("--white-margin", type=int, default=45, help="white line: value this much above road")
    p.add_argument("--white-s", type=int, default=60, help="white line: saturation below this")
    p.add_argument("--color-thr", type=float, default=22, help="weighted-LAB distance to seed that stops a ray")
    p.add_argument("--wl", type=float, default=0.1, help="weight on L (low = ignore track brightness gradient)")
    p.add_argument("--consec", type=int, default=3, help="consecutive stop-pixels before halting (noise guard)")
    p.add_argument("--calib", choices=["global", "frame"], default="global",
                   help="global = one fixed track colour for the run (off-track diverges); frame = per-frame seed")
    p.add_argument("--calib-sample", type=int, default=150, help="frames sampled to estimate the global colour")
    p.add_argument("--offtrack-cov", type=float, default=0.03, help="mask coverage below this = OFF-TRACK banner")
    p.add_argument("--horizon", type=float, default=0.0, help="rays can't climb above this y-fraction (sky guard; 0=off)")
    p.add_argument("--edge-thr", type=float, default=22, help="local LAB jump (vs a few px back) that stops a ray: white line / grass edge")
    p.add_argument("--edge-window", type=int, default=4, help="how many px back the local-edge compares against")
    p.add_argument("--shadow-pass", action="store_true", help="shadow-robust: pass dark+neutral patches (shadows/trees); stop on chroma/bright")
    p.add_argument("--stride", type=int, default=1, help="process every Nth frame when writing masks")
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="ray_preview.png")
    p.add_argument("--video", action="store_true")
    p.add_argument("--vid-out", default="ray_overlay.mp4")
    p.add_argument("--max-frames", type=int, default=1500)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--scale", type=int, default=2)
    return p.parse_args()


def list_imgs(d):
    return [p for p in glob.glob(os.path.join(d, "*")) if p.lower().endswith((".jpg", ".jpeg", ".png"))]


def numeric_key(p):
    h = os.path.basename(p).split("_")[0]
    return int(h) if h.isdigit() else p


def kwargs(a):
    return dict(seed_y=a.seed_y, n_rays=a.n_rays, a0=a.a0, a1=a.a1,
               white_margin=a.white_margin, white_s=a.white_s, color_thr=a.color_thr, consec=a.consec,
               wl=a.wl, horizon=a.horizon, edge_thr=a.edge_thr, edge_window=a.edge_window,
               shadow_pass=a.shadow_pass)


def main():
    a = parse_args()
    paths = sorted(list_imgs(a.img_dir), key=numeric_key)
    if not paths:
        print(f"no images in {a.img_dir}"); return

    extra = {}
    if a.calib == "global":
        gref, grefv = calibrate(paths, a.calib_sample)
        extra = dict(ref=gref, ref_v=grefv)
        print(f"global track ref: LAB({gref[0]:.0f},{gref[1]:.0f},{gref[2]:.0f}) V{grefv:.0f} "
              f"(from {min(a.calib_sample, len(paths))} frames)")

    if a.mask_dir:
        os.makedirs(a.mask_dir, exist_ok=True)
        sel = paths[::a.stride]
        covs = []
        for p in sel:
            mask, _, _ = ray_mask(cv2.imread(p), **kwargs(a), **extra)
            cv2.imwrite(os.path.join(a.mask_dir, os.path.splitext(os.path.basename(p))[0] + ".png"), mask)
            covs.append(mask.mean() / 255 * 100)
        print(f"wrote {len(sel)} ray masks -> {a.mask_dir}/ | coverage mean {np.mean(covs):.1f}%")
        return

    if a.video:
        seq = paths[:a.max_frames] if a.max_frames else paths
        g0 = cv2.imread(seq[0]); H, W = g0.shape[:2]; s = a.scale
        writer = cv2.VideoWriter(a.vid_out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W * s, H * s))
        n_off = 0
        for p in seq:
            bgr = cv2.imread(p)
            mask, eps, seed = ray_mask(bgr, **kwargs(a), **extra)
            cov = mask.mean() / 255
            canvas = cv2.resize(bgr, (W * s, H * s), interpolation=cv2.INTER_LINEAR)
            m = cv2.resize(mask, (W * s, H * s), interpolation=cv2.INTER_NEAREST)
            tint = canvas.copy(); tint[m > 0] = (0, 255, 0)
            canvas = cv2.addWeighted(canvas, 0.6, tint, 0.4, 0)
            for ex, ey in eps:                              # ray endpoints
                cv2.line(canvas, (int(seed[0] * s), int(seed[1] * s)), (int(ex * s), int(ey * s)), (0, 180, 255), 1)
            if cov < a.offtrack_cov:                        # global ref -> off-track collapses coverage
                n_off += 1
                cv2.rectangle(canvas, (0, 0), (W * s - 1, H * s - 1), (0, 0, 255), 6)
                cv2.putText(canvas, "OFF-TRACK", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            writer.write(canvas)
        writer.release()
        print(f"wrote {a.vid_out} ({len(seq)} frames @ {a.fps}fps) | off-track {n_off} ({100*n_off/len(seq):.1f}%)")
        return

    import random
    import matplotlib                                   # lazy: only the preview path needs it
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sel = [paths[i] for i in random.Random(a.seed).sample(range(len(paths)), min(a.n, len(paths)))]
    fig, ax = plt.subplots(len(sel), 2, figsize=(7, 3 * len(sel)))
    if len(sel) == 1:
        ax = ax[None, :]
    for r, p in enumerate(sel):
        bgr = cv2.imread(p); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask, eps, seed = ray_mask(bgr, **kwargs(a), **extra)
        ov = rgb.astype(np.float32) / 255
        ov[mask > 0] = 0.55 * ov[mask > 0] + 0.45 * np.array([0, 1, 0])
        ax[r, 0].imshow(rgb); ax[r, 0].axis("off"); ax[r, 0].set_title("input")
        ax[r, 1].imshow(ov)
        for ex, ey in eps:
            ax[r, 1].plot([seed[0], ex], [seed[1], ey], color="orange", lw=0.4)
        ax[r, 1].set_title(f"ray drivable {mask.mean()/255*100:.0f}%"); ax[r, 1].axis("off")
    plt.tight_layout(); fig.savefig(a.out, dpi=110, bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
