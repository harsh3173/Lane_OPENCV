# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

A **VGG16-BN-encoder UNet** does binary lane segmentation on TuSimple. **Scripts are the primary
workflow** (the notebook proved awkward to run); `lanenet.ipynb` is kept as a reference walkthrough
of the same pipeline. This directory is not a git repository.

## Scripts

Flat module layout (no package), all run via the venv:

- `common.py` — device select (MPS), ImageNet norm/`denorm`, `dice_loss`, `make_criterion` (BCE+Dice), `eval_metrics` (IoU/Dice).
- `dataset.py` — `load_records`, `lane_mask` (polyline→binary mask), `TuSimpleLanes`, `build_train_val_loaders`.
- `model.py` — `VGGUNet` (VGG16-BN features sliced at pool boundaries → UNet decoder; ~20M params).
- `train.py` — entry point; argparse; checkpoints best val-Dice to `vgg_unet_lanenet.pt`.
- `evaluate.py` — load checkpoint, score IoU/Dice on the test split.
- `visualize.py` — headless (matplotlib `Agg`); saves input/GT/pred panels to a **PNG** (no GUI).

Typical commands:

```
.venv/bin/python train.py --epochs 1 --max-samples 100   # fast MVP smoke run
.venv/bin/python train.py --epochs 15 --max-samples 0     # full set (0 = no cap)
.venv/bin/python evaluate.py --ckpt vgg_unet_lanenet.pt --max-samples 200
.venv/bin/python visualize.py --ckpt vgg_unet_lanenet.pt --n 4 --out preds.png
```

All knobs (img size, lane width, batch, lr, paths) are argparse flags — see `--help`. On this M1,
pass `--num-workers 0` if dataloader workers act up. `visualize.py` writes a file rather than
showing a window, so prefer it over notebook display when checking results.

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
