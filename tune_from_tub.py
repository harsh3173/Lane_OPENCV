"""Calibrate the ray-cast pilot's CLASSICAL steering gains against recorded PS4 telemetry.

Interpretation A (see memory no-cte-reward / car-access): the human `user/angle` column is used ONLY
to fit a handful of SCALAR control constants (steer_gain, steer_trim, asymmetric gain_left/right) and
to sanity-check sign convention. It is NOT a per-frame training label, NO model is trained, and the
telemetry is NEVER read while driving — the live pilot still steers purely from the image rays.

What it does:
  1. Loads a DonkeyCar tub (catalog_*.catalog + images/), reading per frame: image, user/angle, user/throttle.
  2. Calibrates the colour ref from the tub images, runs RayPilot.perceive on each frame, and extracts
     the raw, gain-free steering error  err = (90 - heading) / 90.
  3. On MOVING, on-track frames, compares err to the human angle:
        - Pearson correlation + sign-agreement %   (does the ray heading point where the human steered?)
        - least-squares fit  angle ~= gain*err + trim   -> suggested steer_gain / steer_trim
        - separate left/right slopes               -> suggested gain_left / gain_right
  4. Writes a diagnostic PNG (scatter+fit, time series overlay, residual hist) and prints the suggested
     CTRL dict to paste into drive_physical_raycast.py / a profile.

    .venv/bin/python tune_from_tub.py --tub tub_real/data --out tune_real.png
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

from raypilot.ray_mask import calibrate
from raypilot.pilot import RayPilot


def load_tub(tub):
    """Return records [(img_path, user_angle, user_throttle), ...] sorted by _index."""
    img_dir = os.path.join(tub, "images")
    recs = []
    for cat in sorted(glob.glob(os.path.join(tub, "*.catalog"))):
        with open(cat) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                img = os.path.join(img_dir, d["cam/image_array"])
                if os.path.exists(img):
                    recs.append((d["_index"], img, float(d["user/angle"]), float(d["user/throttle"])))
    recs.sort(key=lambda r: r[0])
    return [(img, a, t) for _, img, a, t in recs]


def lstsq_fit(x, y):
    """y ~= m*x + b -> (m, b). Empty/degenerate -> (nan, nan)."""
    if len(x) < 2 or np.ptp(x) < 1e-9:
        return float("nan"), float("nan")
    A = np.vstack([x, np.ones_like(x)]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(m), float(b)


def parse_args():
    p = argparse.ArgumentParser(description="Calibrate ray-pilot steering gains vs recorded PS4 telemetry")
    p.add_argument("--tub", required=True, help="tub dir containing *.catalog + images/")
    p.add_argument("--out", default="tune_tub.png")
    # ray params (defaults = drive_physical_raycast RAY_KW; more rays for offline precision)
    p.add_argument("--n-rays", type=int, default=80)
    p.add_argument("--a0", type=float, default=8)
    p.add_argument("--a1", type=float, default=172)
    p.add_argument("--seed-y", type=float, default=0.85)
    p.add_argument("--white-margin", type=int, default=90)
    p.add_argument("--white-s", type=int, default=60)
    p.add_argument("--color-thr", type=float, default=40)
    p.add_argument("--wl", type=float, default=0.1)
    p.add_argument("--horizon", type=float, default=0.35)
    p.add_argument("--edge-thr", type=float, default=22)
    p.add_argument("--edge-window", type=int, default=4)
    p.add_argument("--weight", choices=["pixel", "ground"], default="ground")
    p.add_argument("--offtrack-cov", type=float, default=0.10)
    p.add_argument("--calib-sample", type=int, default=150)
    # which frames count for the fit
    p.add_argument("--min-throttle", type=float, default=0.02, help="ignore (near-)stopped frames")
    p.add_argument("--turn-deadband", type=float, default=0.10, help="|angle| above this counts for sign-agreement")
    p.add_argument("--max-frames", type=int, default=0, help="0 = all")
    return p.parse_args()


def main():
    a = parse_args()
    recs = load_tub(a.tub)
    if not recs:
        print(f"no records under {a.tub}"); return
    if a.max_frames:
        recs = recs[:a.max_frames]
    paths = [r[0] for r in recs]
    print(f"loaded {len(recs)} frames from {a.tub}")

    ref, ref_v = calibrate(paths, a.calib_sample)
    print(f"global ref LAB({ref[0]:.0f},{ref[1]:.0f},{ref[2]:.0f}) V{ref_v:.0f}")
    ray_kw = dict(seed_y=a.seed_y, n_rays=a.n_rays, a0=a.a0, a1=a.a1, white_margin=a.white_margin,
                  white_s=a.white_s, color_thr=a.color_thr, wl=a.wl, horizon=a.horizon,
                  edge_thr=a.edge_thr, edge_window=a.edge_window)
    # gain 1, trim 0, no EMA/deadband -> r["heading"] is the raw per-frame free-space heading
    pilot = RayPilot(ref, ref_v, ray_kw, steer_gain=1.0, ema=1.0, offtrack_cov=a.offtrack_cov,
                     weight=a.weight)

    err, human, thr, off, cov = [], [], [], [], []
    for img, ang, t in recs:
        bgr = cv2.imread(img)
        r = pilot.perceive(bgr)
        err.append((90.0 - r["heading"]) / 90.0)
        human.append(ang); thr.append(t); off.append(r["offtrack"]); cov.append(r["coverage"])
    err = np.array(err); human = np.array(human); thr = np.array(thr)
    off = np.array(off, bool); cov = np.array(cov)

    move = (thr > a.min_throttle) & (~off) & (cov >= a.offtrack_cov)
    n_move = int(move.sum())
    print(f"\nframes: {len(err)} total | {n_move} moving+on-track used for fit "
          f"| {int(off.sum())} off-track | {int((thr<=a.min_throttle).sum())} stopped")
    if n_move < 10:
        print("too few usable frames to calibrate"); return

    e, h = err[move], human[move]
    corr = float(np.corrcoef(e, h)[0, 1])
    flip = corr < 0
    # sign-agreement on genuine turns
    turn = np.abs(h) > a.turn_deadband
    if turn.sum():
        agree = float((np.sign(e[turn]) == np.sign(h[turn])).mean())
    else:
        agree = float("nan")

    slope, trim = lstsq_fit(e, h)
    gain = abs(slope)
    # asymmetric: split by error sign (matches pilot's g = gl if error<0 else gr)
    left = e < 0; right = e > 0
    ml, _ = lstsq_fit(e[left], h[left]) if left.sum() > 5 else (float("nan"), 0)
    mr, _ = lstsq_fit(e[right], h[right]) if right.sum() > 5 else (float("nan"), 0)

    print("\n=== ray-heading vs human steering (moving, on-track) ===")
    print(f"Pearson corr(err, angle) : {corr:+.3f}   {'(FLIPPED — conventions oppose)' if flip else '(aligned)'}")
    print(f"sign-agreement on turns  : {agree*100:.0f}%  (|angle|>{a.turn_deadband}, n={int(turn.sum())})")
    print(f"least-squares slope/trim : angle ~= {slope:+.2f}*err {trim:+.3f}")
    print(f"left-turn slope          : {ml:+.2f}   right-turn slope: {mr:+.2f}")

    print("\n=== suggested CTRL (interpretation A — scalars only) ===")
    if flip:
        print("  ! correlation is NEGATIVE: swap STEERING_LEFT_PWM/STEERING_RIGHT_PWM (or negate steer).")
    print(f"  steer_gain  = {gain:.2f}")
    print(f"  steer_trim  = {trim:+.3f}")
    if np.isfinite(ml) and np.isfinite(mr):
        print(f"  gain_left   = {abs(ml):.2f}")
        print(f"  gain_right  = {abs(mr):.2f}   (set both only if they differ notably from steer_gain)")

    # ---- diagnostic figure ----
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    ax[0].scatter(e, h, s=6, alpha=0.3)
    xs = np.linspace(e.min(), e.max(), 50)
    ax[0].plot(xs, slope * xs + trim, "r-", lw=2, label=f"{slope:+.2f}x{trim:+.2f}")
    ax[0].axhline(0, color="k", lw=0.5); ax[0].axvline(0, color="k", lw=0.5)
    ax[0].set_xlabel("ray error (90-heading)/90"); ax[0].set_ylabel("human user/angle")
    ax[0].set_title(f"corr {corr:+.2f} | agree {agree*100:.0f}%"); ax[0].legend()

    idx = np.where(move)[0]
    ax[1].plot(idx, h, label="human angle", lw=1)
    ax[1].plot(idx, np.clip(slope * e + trim, -1, 1), label="gain*err+trim", lw=1, alpha=0.8)
    ax[1].set_xlabel("frame"); ax[1].set_title("fit overlay (moving frames)"); ax[1].legend()

    ax[2].hist(h - (slope * e + trim), bins=40)
    ax[2].set_xlabel("residual (human - fit)"); ax[2].set_title("residuals")
    plt.tight_layout(); fig.savefig(a.out, dpi=110)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
