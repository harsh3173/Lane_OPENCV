"""Drivability GOAL flow-field (classical, image-only, no telemetry).

Representation: not a fitted lane curve but a FIELD. The ray drivable mask gives the road corridor;
its far end (where the road exits toward the horizon) is the GOAL. We render a grid of small arrows --
green pointing toward the goal (where to drive) on drivable cells, red dots on non-drivable cells --
and steering falls straight out of it: the heading from the car to the goal IS the steer command
(no curve fit, no path tracing). The goal point is EMA-smoothed over time so the heading doesn't jitter.

Fuses with the ray-cast (the mask is the road); off-track = the corridor collapses / the goal is lost.

    FlowField(ref, ref_v, ray_kw).perceive(bgr) -> {mask, arrows, goal, steer, offtrack, coverage}
"""
import numpy as np
import cv2

from .ray_mask import cast_rays


class FlowPart:
    """DonkeyCar-Part-style adapter around FlowField (mirrors RayPilotPart): run(img)->(angle,throttle).
    Lets drive_gym / a vehicle assembly swap to flow-field steering. Image RGB in (gym/camera), BGR
    inside. FlowField has no throttle model, so we drive a constant throttle (0 when off-track)."""

    def __init__(self, ref, ref_v, ray_kw, const_throttle=0.17, stop_on_offtrack=True, **ff_kw):
        self.flow = FlowField(ref, ref_v, ray_kw, **ff_kw)
        self.const_throttle, self.stop_on_offtrack = const_throttle, stop_on_offtrack
        self.last = (0.0, 0.0)

    def run(self, img_arr):
        if isinstance(img_arr, (tuple, list)):
            img_arr = img_arr[0]
        if img_arr is None:
            return self.last
        arr = np.asarray(img_arr)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return self.last
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        r = self.flow.perceive(bgr)
        self.last_r, self.last_bgr = r, bgr
        angle = float(np.clip(r["steer"], -1.0, 1.0))
        throttle = 0.0 if (r["offtrack"] and self.stop_on_offtrack) else float(self.const_throttle)
        self.last = (angle, throttle)
        return angle, throttle

    def shutdown(self):
        pass


class FlowField:
    def __init__(self, ref, ref_v, ray_kw,
                 band=(0.42, 0.90), grid_rows=6, grid_cols=6, min_span=6,
                 seed_y=0.86, steer_gain=2.6, ema=0.4, goal_ema=0.3,
                 offtrack_cov=0.10, offtrack_on=4, offtrack_off=3, exit_rows=6):
        self.ref, self.ref_v, self.ray_kw = ref, ref_v, ray_kw
        self.band, self.min_span, self.exit_rows = band, min_span, exit_rows
        self.fy = np.linspace(band[0], band[1], grid_rows)     # grid row fractions
        self.fx = np.linspace(0.20, 0.80, grid_cols)           # grid col fractions
        self.seed_y, self.steer_gain, self.ema, self.goal_ema = seed_y, steer_gain, ema, goal_ema
        self.offtrack_cov, self.offtrack_on, self.offtrack_off = offtrack_cov, offtrack_on, offtrack_off
        self.reset()

    def reset(self):
        self.s = 0.0
        self.goal = None                                       # EMA-smoothed goal (corridor exit) x,y
        self._low = self._good = 0
        self.offtrack_state = False

    def _mask(self, bgr):
        H, W = bgr.shape[:2]
        eps, seed, _ = cast_rays(bgr, ref=self.ref, ref_v=self.ref_v, **self.ray_kw)
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [np.array([seed] + eps, np.int32)], 255)
        return m, m.mean() / 255.0

    def perceive(self, bgr):
        H, W = bgr.shape[:2]
        m, cov = self._mask(bgr)
        # per-row road centre -> corridor exit = centroid of the top (farthest) drivable rows
        rows = range(int(self.band[0] * H), int(self.band[1] * H))
        cen = {}
        for y in rows:
            xs = np.where(m[y] > 0)[0]
            if xs.size >= self.min_span:
                cen[y] = 0.5 * (xs.min() + xs.max())
        goal = None
        if cen:
            ys = sorted(cen)                                   # ascending y = far -> near
            top = ys[:self.exit_rows]
            goal = (float(np.mean([cen[y] for y in top])), float(np.mean(top)))
        # temporal EMA on the goal so the heading is steady
        if goal is not None:
            self.goal = goal if self.goal is None else (
                self.goal_ema * np.array(goal) + (1 - self.goal_ema) * np.array(self.goal))
            self.goal = (float(self.goal[0]), float(self.goal[1]))

        # off-track hysteresis (corridor collapsed / goal lost)
        bad = cov < self.offtrack_cov or self.goal is None
        self._low = self._low + 1 if bad else 0
        self._good = 0 if bad else self._good + 1
        if not self.offtrack_state and self._low >= self.offtrack_on:
            self.offtrack_state = True
        elif self.offtrack_state and self._good >= self.offtrack_off:
            self.offtrack_state = False
        offtrack = self.offtrack_state

        # grid of arrows: each drivable cell points toward the goal; non-drivable -> red
        arrows = []
        for fy in self.fy:
            py = int(fy * H)
            for fx in self.fx:
                px = int(fx * W)
                drivable = bool(m[py, min(W - 1, max(0, px))] > 0) and self.goal is not None
                ang = None
                if drivable:
                    ang = float(np.degrees(np.arctan2(py - self.goal[1], self.goal[0] - px)))
                arrows.append((px, py, ang, drivable))

        # steer = heading from the car (seed) to the goal
        if self.goal is not None and not offtrack:
            sx, sy = W / 2.0, self.seed_y * H
            heading = float(np.degrees(np.arctan2(sy - self.goal[1], self.goal[0] - sx)))
            err = (90.0 - heading) / 90.0                      # +ve -> goal is right -> steer right
            raw = float(np.clip(self.steer_gain * err, -1.0, 1.0))
            self.s = self.ema * raw + (1 - self.ema) * self.s
        # else: hold last steer (don't jerk)
        return dict(mask=m, coverage=cov, arrows=arrows, goal=self.goal,
                    steer=float(self.s), offtrack=offtrack)


def draw(bgr, r, scale=4):
    H, W = bgr.shape[:2]
    s = scale
    c = cv2.resize(bgr, (W * s, H * s), interpolation=cv2.INTER_LINEAR)
    mm = cv2.resize(r["mask"], (W * s, H * s), interpolation=cv2.INTER_NEAREST)
    t = c.copy(); t[mm > 0] = (0, 80, 0); c = cv2.addWeighted(c, 0.85, t, 0.15, 0)
    for px, py, ang, drivable in r["arrows"]:
        cx, cy = px * s, py * s
        if not drivable:
            cv2.circle(c, (cx, cy), 4, (0, 0, 255), -1)
        else:
            rad = np.deg2rad(ang); ex, ey = cx + np.cos(rad) * 16, cy - np.sin(rad) * 16
            cv2.arrowedLine(c, (cx, cy), (int(ex), int(ey)), (0, 255, 0), 2, tipLength=0.4)
    if r["goal"] is not None:                                  # the corridor exit (where to drive)
        cv2.circle(c, (int(r["goal"][0] * s), int(r["goal"][1] * s)), 6, (0, 220, 255), 2)
    # big heading arrow from the car to the goal (the actual steer)
    sx, sy = int(W / 2 * s), int(0.88 * H * s)
    ang = np.deg2rad(90.0 - r["steer"] * 60.0)
    cv2.arrowedLine(c, (sx, sy), (int(sx + np.cos(ang) * 70), int(sy - np.sin(ang) * 70)),
                    (0, 0, 255) if r["offtrack"] else (0, 200, 255), 3, tipLength=0.25)
    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(c, f"steer {r['steer']:+.2f}  cov {r['coverage']*100:.0f}%", (8, 24), f, 0.6, (0, 255, 255), 2)
    status = "OFF-TRACK" if r["offtrack"] else "ON-TRACK"
    cv2.putText(c, status, (8, 48), f, 0.6, (0, 0, 255) if r["offtrack"] else (0, 220, 0), 2)
    if r["offtrack"]:
        cv2.rectangle(c, (0, 0), (W * s - 1, H * s - 1), (0, 0, 255), 5)
    return c
