# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

The active project is a **classical, image-only, training-free ray-cast driving pilot** for a
DonkeyCar (sim → real Pi 5). The original VGG16-BN-UNet TuSimple segmentation pipeline still lives in
`legacy/` but is **not used** by the pilot. This directory IS a git repository (code only; tubs,
datasets, videos, images, checkpoints are gitignored).

**Hard constraint:** never use the sim's CTE/reward (or the recorded steering column as a shortcut
label) anywhere in the pilot — perception, control, calibration, or recovery. All signals are
image-derived.

## Layout (package + entry-point scripts)

Library logic is the `raypilot/` package; entry-point scripts stay at the repo root and run via the
venv from the repo root (so `import raypilot...` resolves).

- `raypilot/ray_mask.py` — perception core: `cast_rays` (radial free-space rays), `calibrate`,
  `seed_ref`, `list_imgs`, `numeric_key`. Run-level global colour ref (off-track ⇒ rays collapse).
- `raypilot/pilot.py` — `RayPilot.perceive(bgr) -> {mask, steer, throttle, offtrack, heading, coverage, ...}`
  (free-space-heading steering, EMA, off-track hysteresis); `draw()` overlay; save/load profiles.
- `raypilot/recovery.py` — `RecoveryController`: off-track recovery state machine
  `DRIVE→SLOW→REVERSE→STUCK`. Detects the collapse early on `coverage`, slows, then PULSED reverse
  with steer HELD at last-good (retrace) until the track is re-acquired. Image-only.
- `raypilot/donkey_part.py` — `RayPilotPart` (duck-typed DonkeyCar Part; also used in the gym loop).

Entry points (root):
- `drive_gym.py` — closed-loop sim driver (gym_donkeycar); recovery wired in (`--recovery` on by default).
- `drive_physical_raycast.py` — physical Pi driver (PCA9685 PWM, PiCamera, 3 s straight calibrate → drive).
- `tune_from_tub.py` — calibrate steering gains vs recorded PS4 telemetry (scalars only; never read at drive time).
- `render_overlay.py` — offline overlay video/preview on a folder of frames (was `ray_pilot.py --video`).

Other dirs: `experiments/` (multi_ray, lane_probe — deferred), `legacy/` (the unused VGG-UNet pipeline).

Typical commands:

```
# sim closed loop with recovery (reverse on early off-track, hold-steer retrace):
.venv/bin/python drive_gym.py --profile calib_sim.json --steps 2000 --warn-cov 0.13 --recover-cov 0.15
# offline overlay video:
.venv/bin/python render_overlay.py --img-dir tub_real/data/images --weight ground --steer-gain 3.4 \
    --offtrack-cov 0.10 --video --vid-out pilot_real.mp4 --fps 20
# calibrate gains vs PS4 tub:
.venv/bin/python tune_from_tub.py --tub tub_real/data --out tune_real.png
```

Legacy VGG-UNet pipeline (unused by the pilot) is run from within `legacy/` (flat intra-imports):
`.venv/bin/python legacy/train.py --epochs 15 --max-samples 0`, etc. All knobs are argparse flags
(`--help`); on this M1 pass `--num-workers 0` if dataloader workers act up.

## Environment (use this consistently)

Dedicated venv at `.venv` (Python 3.13, arm64). PyTorch uses the **MPS** backend on this M1.
Registered Jupyter kernel: `Python (vgg-lanenet)` — select it when running the notebook.

- Run a command in the env: `.venv/bin/python ...`, `.venv/bin/pip ...`
- Pinned deps: `requirements.txt` (`.venv/bin/pip install -r requirements.txt` to recreate).
- Stack: torch 2.12 + torchvision 0.27, opencv-python, numpy, matplotlib, pillow, tqdm, jupyter.
- Device line in the notebook auto-selects `mps`; there is no CUDA/AMP here.
- **Disk is near-full** (~3 GB free). `pip install` without `--no-cache-dir` previously filled the
  disk — always pass `--no-cache-dir`, and purge `~/Library/Caches/pip` if installs fail with
  `No space left on device`.

## Data layout

TUSimple Lane Detection Challenge data, split across two locations:

- `TUSimple/train_set/label_data_{0313,0531,0601}.json` — training labels (one JSON object per line).
- `TUSimple/test_set/test_tasks_0627.json` — test submission template (lanes empty, fill predictions in).
- `TUSimple/test_label.json` and `test_label_new.json` (repo root) — test ground-truth labels.

The image clips **are** present: `TUSimple/train_set/clips/` (3626 clips) and
`TUSimple/test_set/clips/` (2782 clips), each clip a folder of 20 frames (`1.jpg`…`20.jpg`).
`raw_file` paths are relative to the split folder — i.e. join with `train_set/` or `test_set/`,
**not** the repo root. Labels annotate only the 20th frame.

## Label format (critical to get right)

Each line is one JSON object describing the **last (20th) frame** of a clip:

```
{
  "raw_file":  "clips/0313/<seq>/20.jpg",   // image path
  "lanes":     [[x, x, ...], ...],          // per-lane x pixel values
  "h_samples": [240, 250, ..., 710]         // y pixel values, shared by all lanes in that object
}
```

Key conventions:
- A lane is a polyline. The i-th point of a lane is `(lanes[k][i], h_samples[i])` — `lanes` gives x, `h_samples` gives y. They are index-paired, so `len(lanes[k]) == len(h_samples)`.
- `-2` means "no lane marking at this y" — must be masked out, not drawn or used as a coordinate.
- At most 5 lanes per frame; the model is expected to predict at most 4 (current + left/right).
- `h_samples` is **not** constant across files: train labels use 240–710 (step 10); `test_label_new.json` and `test_tasks_0627.json` use 160–710. Do not hard-code the y-range — read `h_samples` per object.
- Test prediction format adds `run_time` (ms per frame) and omits `h_samples` in the output (predictions must align to the template's `h_samples`).

## Working in the notebook

Run with Jupyter (`jupyter lab lanenet.ipynb`) or the VS Code notebook UI. There is no `requirements.txt`/`environment.yml` yet — when adding a deep-learning stack, pin it in a new environment file rather than relying on the ambient interpreter, and confirm the framework (PyTorch vs TensorFlow) with the user before scaffolding, since neither is established here.
