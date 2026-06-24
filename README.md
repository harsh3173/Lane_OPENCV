# Ray-Cast Drivable Pilot

A **classical, image-only, label-free** autonomous driving pipeline for the DonkeyCar simulator
(and real car). It detects the drivable surface by casting rays from the camera's vanishing point,
then steers toward the most open free space — **no neural network, no training, and no CTE/reward
ever used**. It runs at ~45 FPS offline and drives the DonkeyCar `generated_track` continuously
(>1000 steps / a full lap) in closed loop.

> Origin: this project began as a VGG16-UNet lane segmenter on TuSimple (legacy scripts
> `common/dataset/model/train/evaluate/visualize.py`, see `CLAUDE.md`). The CNN did not transfer
> across domains; the ray-cast approach below replaced it and generalizes across the sim road, an
> indoor mat track, and a warehouse floor with only a few per-domain knobs.

## How it works

```
camera frame ──► cast_rays ──► drivable mask + ray endpoints ──► free-space heading ──► steer/throttle
```

1. **Radial ray-cast** (`ray_mask.cast_rays`) — fan N rays from the bottom-centre (the seed, lifted
   above the hood). Each ray marches outward until a STOP:
   - **white lane line** — low saturation AND value > road + margin (relative, so bright road isn't
     mistaken for a line),
   - **off-colour edge** — weighted-LAB distance to a fixed reference (L down-weighted so the road's
     own brightness gradient doesn't stop the ray; grass/floor chroma does),
   - **local edge** — a sharp LAB jump vs a few px back (catches lines / boundaries while the road's
     smooth gradient passes),
   - **horizon cap** — rays can't climb above the horizon (sky guard).
   The swept polygon of endpoints is the **drivable mask**.

2. **Global colour calibration** (`ray_mask.calibrate`) — one fixed track colour for the whole run
   (median of seed patches). It does **not** drift, so when the car goes off-track the live colour
   diverges from it, the rays collapse, and coverage → 0 = **off-track**.

3. **Free-space steering** (`ray_pilot.RayPilot`) — each ray's free length (or perspective-corrected
   ground distance, `--weight ground`) weights its angle; the resulting heading vs straight-up gives
   steer ∈ [-1, 1]. Throttle scales with forward clearance. Both are EMA-smoothed.

### Robustness fixes (learned in closed loop)
- **`yellow_pass`** — the yellow centre line is *drivable*; rays pass through it so crossing/
  straddling the line doesn't collapse coverage into a false off-track.
- **Off-track hysteresis + hold-steer-on-collapse** — a 1–2 frame collapse (yellow line, checkered
  start) is ignored; steering holds its last value instead of jerking to zero.
- **Live startup calibration** — creep forward, sample the *real* road colour, lock the reference
  (no dependence on a pre-recorded tub).
- **Constant-throttle mode** — keeps the car moving so it never deadlocks on the start line.

## Files (code)

| File | Role |
|---|---|
| `ray_mask.py` | core ray-cast: `cast_rays`, `calibrate`, drivable mask; CLI for preview / mask-dir / overlay video |
| `ray_pilot.py` | `RayPilot.perceive(bgr) → {mask, steer, throttle, offtrack, …}`; calibration profiles |
| `donkey_part.py` | `RayPilotPart` — DonkeyCar Part (RGB frame → angle, throttle), duck-typed |
| `drive_gym.py` | closed-loop driver for `gym_donkeycar` (gymnasium); live calibration; overlay recording |
| `ray_pilot_ROADMAP.md` | sim→real roadmap |
| `common/dataset/model/train/evaluate/visualize.py` | legacy VGG-UNet lane segmenter (not used by the pilot) |

## Usage

### Offline (preview / video / masks) — uses the repo `.venv`
```bash
.venv/bin/python ray_mask.py --img-dir <frames> --n 6 --out preview.png       # preview
.venv/bin/python ray_mask.py --img-dir <frames> --video --vid-out rays.mp4     # overlay video
```

### Closed-loop in the DonkeyCar sim — uses the `donkey` conda env
Requires `gymnasium` + `gym_donkeycar` + the DonkeySim binary.
```bash
# 1) one-off calibration profile from a recorded tub of the same track:
python ray_pilot.py --img-dir <tub> --weight ground --horizon 0.35 \
    --save-profile calib_sim.json --max-frames 1
# 2) drive (live-calibrates on startup, constant throttle, records an overlay):
python drive_gym.py --profile calib_sim.json --env donkey-generated-track-v0 \
    --steer-gain 3.0 --weight ground --const-throttle 0.2 --record run.mp4
```

### Physical car (Raspberry Pi + DonkeyCar hardware)
`drive_physical_raycast.py` — same hardware scaffold as a TFLite donkey driver (PiCamera 128×120,
PCA9685 PWM, safety shutdown) but runs `RayPilot.perceive()` instead of a model. On boot it does a
~3 s straight calibration (creep straight, sample the road, lock the colour ref), then drives.
Image-only, no model file — works on a fresh clone. Tune `STEERING_GAIN`/`THROTTLE_BOOST`/PWM
constants at the top; the runtime needs only `cv2` + `numpy` (matplotlib is import-lazy).
```bash
python drive_physical_raycast.py     # calibrate 3s straight on track, then autonomous
```

### Per-domain ray params
| Domain | Track cue | Key flags |
|---|---|---|
| sim road | grey road, yellow/white lines | `--white-margin 90 --color-thr 40 --wl 0.1 --horizon 0.35 --edge-thr 22` |
| warehouse | tan floor, dark shelving | `--horizon 0.35 --edge-thr 18 --color-thr 45 --a0 22 --a1 158` |
| old-car | dark mat, light floor | defaults |

## Result
generated_track, image-only, `--weight ground --steer-gain 3.0 --const-throttle 0.17`:
**1 reset in 1400 steps, ~858 steps continuous (~43 s)**, steer smoothness (std Δsteer) ≈ 0.055.

Anti-oscillation knobs (optional): `--deadband` (ignore tiny heading errors → no weave on straights),
`--steer-damp` (PD damping), lower `--ema`/throttle for smoother/slower. In practice the dominant weave
came from a perceive-twice-per-step recording bug (now fixed); the plain baseline drives smoothly.

## Constraints (by design)
- **No NN, no training** — purely geometric/classical computation.
- **No CTE or reward** is ever used (not for control, tuning targets, or anything trained).
