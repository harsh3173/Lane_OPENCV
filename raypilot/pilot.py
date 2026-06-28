"""Phase 0: live perception + control from the ray-cast (image-only, no telemetry).

Wraps the ray-cast core (ray_mask.cast_rays) into a per-frame controller:
    perceive(bgr) -> {mask, endpoints, seed, offtrack, steer, throttle, heading, coverage}

Steering = FREE-SPACE HEADING: each ray's length (free distance) weights its angle, so the chosen
heading points where the track is most open -- this follows curves even when the fan mask doesn't
wrap them. Throttle scales with forward clearance and goes to 0 off-track. Steer/throttle are EMA-
smoothed.

Library module (no CLI). The offline video/preview tool lives in render_overlay.py.
"""
import json

import cv2
import numpy as np

from .ray_mask import cast_rays


class RayPilot:
    def __init__(self, ref, ref_v, ray_kw, steer_gain=1.6, base_throttle=0.5,
                 ema=0.4, offtrack_cov=0.03, clear_ref=0.75,
                 weight="pixel", persp_horizon=0.35, min_gap_frac=0.05,
                 offtrack_on=4, offtrack_off=3, steer_deadband=0.0, steer_damp=0.0,
                 steer_trim=0.0, gain_left=None, gain_right=None):
        self.ref, self.ref_v, self.ray_kw = ref, ref_v, ray_kw
        self.steer_gain, self.base_throttle = steer_gain, base_throttle
        self.ema, self.offtrack_cov, self.clear_ref = ema, offtrack_cov, clear_ref
        # heading weighting: "pixel" = ray pixel length; "ground" = perspective-corrected ground
        # distance CLEARED (1/(y-horizon) at endpoint minus at seed) -> vertical rays reach farther
        # in the world per pixel, so they get more vote; horizontal rays ~0. Capped near horizon.
        self.weight, self.persp_horizon, self.min_gap_frac = weight, persp_horizon, min_gap_frac
        # off-track hysteresis: only flag after offtrack_on consecutive low-coverage frames, clear
        # after offtrack_off good ones -> a momentary collapse (crossing the yellow/start line) is ignored
        self.offtrack_on, self.offtrack_off = offtrack_on, offtrack_off
        self._low = self._good = 0
        self.offtrack_state = False
        # anti-oscillation: deadband ignores tiny heading errors (no weave on straights); damp is a
        # derivative term that opposes fast steering swings (smooths curve overshoot)
        self.steer_deadband, self.steer_damp = steer_deadband, steer_damp
        # left/right asymmetry correction: steer_trim = constant center-bias offset (camera/center
        # offset); gain_left/right = per-direction gain (None -> use steer_gain) for unequal turn radii
        self.steer_trim, self.gain_left, self.gain_right = steer_trim, gain_left, gain_right
        self._prev_err = 0.0
        self.s = 0.0          # EMA steer state
        self.t = 0.0          # EMA throttle state
        a0, a1, n = ray_kw["a0"], ray_kw["a1"], ray_kw["n_rays"]
        self.angles = np.linspace(a0, a1, n)

    def perceive(self, bgr):
        H, W = bgr.shape[:2]
        eps, seed, _ = cast_rays(bgr, ref=self.ref, ref_v=self.ref_v, **self.ray_kw)
        ep = np.asarray(eps, np.float32)
        poly = np.array([seed] + eps, np.int32)
        mask = np.zeros((H, W), np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        cov = mask.mean() / 255.0

        lengths = np.hypot(ep[:, 0] - seed[0], ep[:, 1] - seed[1])
        # heading weight per ray
        if self.weight == "ground":
            yh = self.persp_horizon * H
            mg = self.min_gap_frac * H
            d_end = 1.0 / np.maximum(ep[:, 1] - yh, mg)          # ground distance to endpoint
            d_seed = 1.0 / max(seed[1] - yh, mg)
            w = np.clip(d_end - d_seed, 0.0, None)               # ground distance CLEARED
        else:
            w = lengths
        # off-track via hysteresis (debounced) so a 1-2 frame collapse isn't a false off-track
        if cov < self.offtrack_cov:
            self._low += 1; self._good = 0
        else:
            self._good += 1; self._low = 0
        if not self.offtrack_state and self._low >= self.offtrack_on:
            self.offtrack_state = True
        elif self.offtrack_state and self._good >= self.offtrack_off:
            self.offtrack_state = False
        offtrack = self.offtrack_state

        valid = w.sum() > 1e-6 and cov >= self.offtrack_cov     # enough free space to trust a heading
        if valid:
            heading = float((w * self.angles).sum() / w.sum())  # free-space heading
            error = (90.0 - heading) / 90.0                     # +ve = steer right
            if self.steer_deadband > 0:                         # soft deadband -> no weave on straights
                error = (1.0 if error > 0 else -1.0) * max(0.0, abs(error) - self.steer_deadband)
            de = error - self._prev_err                         # PD damping term
            self._prev_err = error
            gl = self.steer_gain if self.gain_left is None else self.gain_left
            gr = self.steer_gain if self.gain_right is None else self.gain_right
            g = gl if error < 0 else gr                         # per-direction gain (asymmetry)
            raw_steer = float(np.clip(g * error + self.steer_damp * de + self.steer_trim, -1.0, 1.0))
            clearance = np.clip(lengths.max() / (self.clear_ref * H), 0.0, 1.0)
            raw_thr = self.base_throttle * clearance
            self.s = self.ema * raw_steer + (1 - self.ema) * self.s
            self.t = self.ema * raw_thr + (1 - self.ema) * self.t
        else:
            # collapsed view (crossing a line / truly off): HOLD last steer, don't jerk to 0
            heading, raw_steer = 90.0, self.s
            self.t = 0.0 if offtrack else self.t
        return dict(mask=mask, endpoints=ep, seed=seed, offtrack=offtrack, heading=heading,
                    coverage=cov, steer=float(self.s), throttle=float(self.t),
                    raw_steer=float(raw_steer))

    # ---- calibration profiles (Phase 1): compute once, reload instantly at runtime ----
    def save_profile(self, path):
        prof = dict(ref=np.asarray(self.ref).tolist(), ref_v=self.ref_v, ray_kw=self.ray_kw,
                    ctrl=dict(steer_gain=self.steer_gain, base_throttle=self.base_throttle,
                              ema=self.ema, offtrack_cov=self.offtrack_cov, clear_ref=self.clear_ref,
                              weight=self.weight, persp_horizon=self.persp_horizon,
                              min_gap_frac=self.min_gap_frac, steer_trim=self.steer_trim,
                              gain_left=self.gain_left, gain_right=self.gain_right))
        with open(path, "w") as f:
            json.dump(prof, f, indent=2)
        return path

    @classmethod
    def from_profile(cls, path):
        with open(path) as f:
            p = json.load(f)
        return cls(np.array(p["ref"], np.float32), p["ref_v"], p["ray_kw"], **p["ctrl"])


def draw(bgr, r, scale=3):
    H, W = bgr.shape[:2]
    s = scale
    canvas = cv2.resize(bgr, (W * s, H * s), interpolation=cv2.INTER_LINEAR)
    m = cv2.resize(r["mask"], (W * s, H * s), interpolation=cv2.INTER_NEAREST)
    tint = canvas.copy(); tint[m > 0] = (0, 255, 0)
    canvas = cv2.addWeighted(canvas, 0.65, tint, 0.35, 0)
    sx, sy = r["seed"]
    # steer arrow: heading derived from the smoothed steer (so the arrow shows the command)
    ang = np.deg2rad(90.0 - r["steer"] / max(1e-6, 1.0) * 60.0)   # +-60deg visual swing
    L = (0.45 * H) * s
    ex, ey = sx * s + np.cos(ang) * L, sy * s - np.sin(ang) * L
    off = r["offtrack"]
    color = (0, 0, 255) if off else (0, 200, 255)
    cv2.arrowedLine(canvas, (int(sx * s), int(sy * s)), (int(ex), int(ey)), color, 3, tipLength=0.2)
    if off:                                              # red border emphasis when off-track
        cv2.rectangle(canvas, (0, 0), (W * s - 1, H * s - 1), (0, 0, 255), 5)
    f = cv2.FONT_HERSHEY_SIMPLEX
    # line 1 (always): steer / throttle
    cv2.putText(canvas, f"steer {r['steer']:+.2f}  thr {r['throttle']:.2f}", (8, 22), f, 0.6, (0, 255, 255), 2)
    # line 2 (always): persistent ON/OFF-TRACK flag + coverage
    status = "OFF-TRACK" if off else "ON-TRACK"
    scol = (0, 0, 255) if off else (0, 220, 0)
    cv2.putText(canvas, f"{status}  cov {r['coverage'] * 100:.0f}%", (8, 46), f, 0.6, scol, 2)
    # line 3 (optional): recovery state machine status (DRIVE/SLOW/REVERSE/STUCK)
    rs = r.get("recovery")
    if rs and rs != "DRIVE":
        rcol = {"SLOW": (0, 200, 255), "STOP": (0, 165, 255), "REVERSE": (0, 0, 255),
                "STUCK": (0, 0, 200)}.get(rs, (255, 255, 255))
        cv2.putText(canvas, f"RECOVERY: {rs}", (8, 70), f, 0.6, rcol, 2)
    return canvas

