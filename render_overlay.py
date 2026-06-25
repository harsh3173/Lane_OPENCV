"""Offline ray-pilot overlay: render the perception+steering overlay on a folder of frames.

Was ray_pilot.py's __main__; now a thin CLI over the raypilot package. Image-only, no telemetry.

    .venv/bin/python render_overlay.py --img-dir tub_real/data/images --weight ground \
        --steer-gain 3.4 --white-margin 90 --color-thr 40 --wl 0.1 --horizon 0.35 \
        --edge-thr 22 --offtrack-cov 0.10 --video --vid-out pilot_real.mp4 --fps 20

    # or load a saved calibration profile instead of recalibrating:
    .venv/bin/python render_overlay.py --img-dir tub_generated_track --profile calib_sim.json --video
"""
import argparse
import time

import cv2
import numpy as np

from raypilot.ray_mask import calibrate, list_imgs, numeric_key
from raypilot.pilot import RayPilot, draw


def parse_args():
    p = argparse.ArgumentParser(description="Offline ray-pilot perception+steering overlay")
    p.add_argument("--img-dir", required=True)
    p.add_argument("--n-rays", type=int, default=80)
    p.add_argument("--a0", type=float, default=8)
    p.add_argument("--a1", type=float, default=172)
    p.add_argument("--seed-y", type=float, default=0.85)
    p.add_argument("--white-margin", type=int, default=45)
    p.add_argument("--white-s", type=int, default=60)
    p.add_argument("--color-thr", type=float, default=40)
    p.add_argument("--wl", type=float, default=0.15)
    p.add_argument("--horizon", type=float, default=0.0)
    p.add_argument("--edge-thr", type=float, default=22)
    p.add_argument("--edge-window", type=int, default=4)
    p.add_argument("--weight", choices=["pixel", "ground"], default="pixel",
                   help="heading weight: pixel length, or perspective-corrected ground distance cleared")
    p.add_argument("--persp-horizon", type=float, default=0.35, help="ground weight: perspective horizon y-fraction")
    p.add_argument("--min-gap-frac", type=float, default=0.05, help="ground weight: caps weight near the horizon")
    p.add_argument("--steer-gain", type=float, default=1.6)
    p.add_argument("--base-throttle", type=float, default=0.5)
    p.add_argument("--ema", type=float, default=0.4)
    p.add_argument("--offtrack-cov", type=float, default=0.03)
    p.add_argument("--calib-sample", type=int, default=150)
    p.add_argument("--profile", default=None, help="load a saved calibration profile (skip calibrate)")
    p.add_argument("--save-profile", default=None, help="write the calibrated profile to this path")
    p.add_argument("--max-frames", type=int, default=1500)
    p.add_argument("--video", action="store_true")
    p.add_argument("--vid-out", default="pilot_overlay.mp4")
    p.add_argument("--fps", type=int, default=15)
    return p.parse_args()


def main():
    a = parse_args()
    paths = sorted(list_imgs(a.img_dir), key=numeric_key)
    if not paths:
        print("no images"); return

    if a.profile:
        pilot = RayPilot.from_profile(a.profile)
        print(f"loaded profile {a.profile} | ref V{pilot.ref_v:.0f} weight={pilot.weight}")
    else:
        ref, ref_v = calibrate(paths, a.calib_sample)
        print(f"global ref LAB({ref[0]:.0f},{ref[1]:.0f},{ref[2]:.0f}) V{ref_v:.0f}")
        ray_kw = dict(seed_y=a.seed_y, n_rays=a.n_rays, a0=a.a0, a1=a.a1,
                      white_margin=a.white_margin, white_s=a.white_s, color_thr=a.color_thr,
                      wl=a.wl, horizon=a.horizon, edge_thr=a.edge_thr, edge_window=a.edge_window)
        pilot = RayPilot(ref, ref_v, ray_kw, a.steer_gain, a.base_throttle, a.ema, a.offtrack_cov,
                         weight=a.weight, persp_horizon=a.persp_horizon, min_gap_frac=a.min_gap_frac)
        if a.save_profile:
            print(f"saved profile -> {pilot.save_profile(a.save_profile)}")

    seq = paths[:a.max_frames] if a.max_frames else paths
    writer = None
    if a.video:
        H, W = cv2.imread(seq[0]).shape[:2]
        writer = cv2.VideoWriter(a.vid_out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W * 3, H * 3))

    t0, n_off, steer_abs = time.time(), 0, []
    for p in seq:
        bgr = cv2.imread(p)
        r = pilot.perceive(bgr)
        n_off += r["offtrack"]; steer_abs.append(abs(r["steer"]))
        if writer is not None:
            writer.write(draw(bgr, r))
    dt = time.time() - t0
    if writer is not None:
        writer.release(); print(f"wrote {a.vid_out}")
    fps = len(seq) / dt
    print(f"processed {len(seq)} frames in {dt:.1f}s -> {fps:.0f} FPS ({1000/fps:.1f} ms/frame, "
          f"perception+control)")
    print(f"off-track {n_off} ({100*n_off/len(seq):.1f}%) | mean |steer| {np.mean(steer_abs):.2f}")


if __name__ == "__main__":
    main()
