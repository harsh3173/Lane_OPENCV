"""Adjacent-lane probes — a centre drivable fan plus two side probes placed HIGHER (further ahead)
and offset left/right so they normally sit off the current lane. Each probe is classified by colour
against the centre-lane reference:
    road-coloured  -> there IS a lane/road on that side  -> cast a fan, colour it as another lane
    different      -> that side is a boundary / off-track -> red marker

Standalone experiment; does NOT modify ray_mask.py / ray_pilot.py / multi_ray.py.

    .venv/bin/python lane_probe.py --img-dir tub_generated_track --n 6 --out probe_preview.png
    .venv/bin/python lane_probe.py --img-dir tub_old_car/images --video --vid-out probe_oldcar.mp4
"""
import argparse
import math

import cv2
import numpy as np

from ray_mask import calibrate, list_imgs, numeric_key   # read-only reuse
from multi_ray import cast_fan, fan_mask                 # read-only reuse

CENTER_COL = (0, 255, 0)
LANE_COLS = [(255, 120, 0), (0, 200, 255)]               # detected adjacent lanes (L, R)
OFF_COL = (0, 0, 255)


def probe_dist(lab, xp, yp, ref, wl, half=4):
    H, W = lab.shape[:2]
    y0, y1 = max(0, yp - half), min(H, yp + half)
    x0, x1 = max(0, xp - half), min(W, xp + half)
    c = np.median(lab[y0:y1, x0:x1].reshape(-1, 3), axis=0)
    wvec = np.array([wl, 1.0, 1.0], np.float32)
    return float(np.sqrt(np.sum(wvec * (c - ref) ** 2)))


def analyze(bgr, ref, ref_v, kw, probes, probe_thr, connect_thr, spread):
    """Return center fan mask, and per-probe (xp, yp, state, dist, fan_mask). state is:
        OFF  - not road-coloured (boundary / off-track)
        SAME - its inward (toward-centre) fan REACHES the centre -> clear road path -> same surface
        LANE - road-coloured but its inward fan is BLOCKED before the centre -> a divider -> new lane

    Side probes cast DOWN-and-INWARD (aimed at the centre point), so the rays directly traverse the
    region between the probe and the centre -> placement-independent connectivity / segregation.
    """
    H, W = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    def fan(sx, sy, a0, a1):
        eps = cast_fan(lab, hsv, ref, ref_v, sx, sy, kw["n_rays"], a0, a1,
                       kw["white_margin"], kw["white_s"], kw["color_thr"], kw["wl"], kw["horizon"],
                       kw["edge_thr"], kw["edge_window"], kw["yellow_pass"])
        return fan_mask((H, W), (sx, sy), eps)

    cx, by = W / 2.0, kw["seed_y"] * H
    center = fan(cx, by, kw["a0"], kw["a1"])               # centre still fans UP (current lane ahead)
    cset = center > 0
    out = []
    for (xf, yf) in probes:
        xp, yp = int(xf * W), int(yf * H)
        d = probe_dist(lab, xp, yp, ref, kw["wl"])
        if d >= probe_thr:
            out.append((xp, yp, "OFF", d, None)); continue
        aim = math.degrees(math.atan2(-(by - yp), (cx - xp)))   # point toward the centre (down-inward)
        inward = fan(float(xp), float(yp), aim - spread, aim + spread)
        overlap = np.logical_and(cset, inward > 0).sum() / max(int((inward > 0).sum()), 1)
        state = "SAME" if overlap >= connect_thr else "LANE"    # reached centre -> same surface
        out.append((xp, yp, state, d, inward))
    return center, out


def overlay(bgr, center, probes_out, scale=3):
    H, W = bgr.shape[:2]; s = scale
    canvas = cv2.resize(bgr, (W * s, H * s), interpolation=cv2.INTER_LINEAR)

    def tint(mask, color, a=0.32):
        mm = cv2.resize(mask, (W * s, H * s), interpolation=cv2.INTER_NEAREST)
        t = canvas.copy(); t[mm > 0] = color
        return cv2.addWeighted(canvas, 1 - a, t, a, 0)

    canvas = tint(center, CENTER_COL)
    labels = []
    for i, (xp, yp, state, d, lane_mask) in enumerate(probes_out):
        side = "L" if i == 0 else "R"
        if state == "LANE":                                  # genuine separate lane -> colour its fan
            canvas = tint(lane_mask, LANE_COLS[i % len(LANE_COLS)]); col = LANE_COLS[i % len(LANE_COLS)]
        elif state == "SAME":                                # same wide surface -> faint, white dot
            canvas = tint(lane_mask, (220, 220, 220), a=0.15); col = (255, 255, 255)
        else:
            col = OFF_COL
        cv2.circle(canvas, (int(xp * s), int(yp * s)), 7, col, -1)
        cv2.circle(canvas, (int(xp * s), int(yp * s)), 7, (0, 0, 0), 1)
        labels.append(f"{side}:{state}({d:.0f})")
    n_lanes = 1 + sum(1 for p in probes_out if p[2] == "LANE")
    cv2.putText(canvas, f"lanes {n_lanes} | " + "  ".join(labels), (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return canvas


def parse_args():
    p = argparse.ArgumentParser(description="Adjacent-lane probes via colour match to centre lane")
    p.add_argument("--img-dir", required=True)
    p.add_argument("--probes", type=float, nargs="+", default=[0.15, 0.50, 0.85, 0.50],
                   help="flat list x,y,x,y... for side probes (smaller y = higher/further ahead)")
    p.add_argument("--probe-thr", type=float, default=22, help="LAB dist to centre ref: below = road-coloured")
    p.add_argument("--probe-spread", type=float, default=50, help="half-angle of the inward fan (deg)")
    p.add_argument("--connect-thr", type=float, default=0.05,
                   help="probe-fan overlap with centre above this = SAME surface; below = separate LANE")
    p.add_argument("--seed-y", type=float, default=0.85)
    p.add_argument("--n-rays", type=int, default=50)
    p.add_argument("--a0", type=float, default=8)
    p.add_argument("--a1", type=float, default=172)
    p.add_argument("--white-margin", type=int, default=90)
    p.add_argument("--white-s", type=int, default=60)
    p.add_argument("--color-thr", type=float, default=40)
    p.add_argument("--wl", type=float, default=0.1)
    p.add_argument("--horizon", type=float, default=0.35)
    p.add_argument("--edge-thr", type=float, default=22)
    p.add_argument("--edge-window", type=int, default=4)
    p.add_argument("--no-yellow-pass", dest="yellow_pass", action="store_false", default=False)
    p.add_argument("--yellow-pass", dest="yellow_pass", action="store_true")
    p.add_argument("--calib-sample", type=int, default=150)
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="probe_preview.png")
    p.add_argument("--video", action="store_true")
    p.add_argument("--vid-out", default="probe_overlay.mp4")
    p.add_argument("--max-frames", type=int, default=1200)
    p.add_argument("--fps", type=int, default=20)
    return p.parse_args()


def main():
    a = parse_args()
    paths = sorted(list_imgs(a.img_dir), key=numeric_key)
    if not paths:
        print("no images"); return
    ref, ref_v = calibrate(paths, a.calib_sample)
    pr = a.probes
    probes = [(pr[i], pr[i + 1]) for i in range(0, len(pr) - 1, 2)]
    print(f"global ref LAB({ref[0]:.0f},{ref[1]:.0f},{ref[2]:.0f}) V{ref_v:.0f} | probes {probes} "
          f"| probe_thr {a.probe_thr}")
    kw = dict(seed_y=a.seed_y, n_rays=a.n_rays, a0=a.a0, a1=a.a1, white_margin=a.white_margin,
              white_s=a.white_s, color_thr=a.color_thr, wl=a.wl, horizon=a.horizon,
              edge_thr=a.edge_thr, edge_window=a.edge_window, yellow_pass=a.yellow_pass)

    if a.video:
        seq = paths[:a.max_frames] if a.max_frames else paths
        H, W = cv2.imread(seq[0]).shape[:2]
        writer = cv2.VideoWriter(a.vid_out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W * 3, H * 3))
        lane_hist = []
        for p in seq:
            bgr = cv2.imread(p)
            center, po = analyze(bgr, ref, ref_v, kw, probes, a.probe_thr, a.connect_thr, a.probe_spread)
            lane_hist.append(1 + sum(1 for x in po if x[2] == "LANE"))
            writer.write(overlay(bgr, center, po))
        writer.release()
        print(f"wrote {a.vid_out} ({len(seq)} frames) | avg lanes detected {np.mean(lane_hist):.2f}")
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
        center, po = analyze(bgr, ref, ref_v, kw, probes, a.probe_thr, a.connect_thr, a.probe_spread)
        ov = cv2.cvtColor(overlay(bgr, center, po, scale=1), cv2.COLOR_BGR2RGB)
        ax[r, 0].imshow(rgb); ax[r, 0].axis("off"); ax[r, 0].set_title("input")
        ax[r, 1].imshow(ov); ax[r, 1].axis("off")
        nlanes = 1 + sum(1 for x in po if x[2] == "LANE")
        ax[r, 1].set_title(f"lanes {nlanes}")
    plt.tight_layout(); fig.savefig(a.out, dpi=110, bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
