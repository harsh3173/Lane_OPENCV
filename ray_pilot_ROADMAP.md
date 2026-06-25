# Ray-Pilot Roadmap — offline masks → live DonkeyCar (sim → real)

Porting `ray_mask.py` (radial free-space ray-cast) from offline tub analysis into a live, image-only
autonomous driving controller. No CTE/telemetry is ever used (see memory: no-cte-reward).

## STATUS (updated)
**Phases 0–2 DONE — the pilot drives the DonkeyCar sim `generated_track` closed-loop, image-only.**
Latest run: **0 resets / 1200 steps, 1174 steps continuous (~59 s)**, steer smoothness std(Δsteer) ≈
0.045, ~20 Hz. Classical, no NN, no CTE/reward (audited & enforced in code).

Pieces in place:
- `ray_mask.py` — rays from bottom-centre stop at white lines (relative-bright) / colour edges /
  local edges; **global colour calibration** (off-track ⇒ rays collapse) + **horizon cap** + **yellow-pass**
  (yellow centre line is drivable). Validated on sim road, old-car mat, warehouse floor.
- `ray_pilot.py` — `RayPilot.perceive → {mask, steer, throttle, offtrack, coverage}`; free-space-heading
  steering (`--weight ground` perspective option), EMA + optional deadband/PD-damp; off-track hysteresis
  + hold-steer-on-collapse; calibration profiles; overlay with persistent ON/OFF-TRACK flag.
- `donkey_part.py` / `drive_gym.py` — DonkeyCar Part + gymnasium closed-loop driver with **live startup
  calibration** (creep, sample real road, lock ref), const-throttle mode, `std Δsteer` metric.

Best config: `--weight ground --steer-gain 3.0 --const-throttle 0.17 --offtrack-cov 0.10`.

Known limit: the fixed-origin symmetric fan can't fully wrap a *sharp* curve in the **mask** — handled
adequately by the free-space-direction controller, but the mid-field/look-ahead weighting refinement
(below) is the remaining lever.

---

## Phase 0 — Core perception+control module (offline) ✅ DONE
45 FPS / 22 ms per frame; free-space-heading steering validated by steer-arrow overlay (correct on
straights). Perspective-weighted (`ground`) heading added per the far-rays-reach-farther insight.
**Goal:** turn the mask into a (steer, throttle) command in a fast, reusable, dependency-light module.
- Extract `ray_perception(image, cfg) -> {mask, endpoints, offtrack, steer, throttle}` (drop matplotlib/CLI/video).
- **Steering law** (pick by experiment): free-space *balance* (left vs right free area) or longest-free-ray
  heading → normalised steer [-1,1]. Prefer free-space heading over mask-centroid — it handles curves.
- **Throttle:** base value, scaled by forward free distance; → 0 (or reverse/stop) when off-track.
- **Temporal smoothing:** EMA on steer (we already proved this kills jitter).
- **Validate offline:** overlay the predicted steer arrow on the existing tub videos; arrow should track the road.
- **Perf:** profile the ray loop; target <30–50 ms/frame on dev. Lever: fewer rays, vectorise the march.
- **DoD:** steer arrow follows the road on sim+warehouse tub videos; meets FPS target.

## Phase 1 — Live calibration ✅ DONE
`drive_gym` creeps forward at start, samples the live road, locks the global ref (matched the
recorded ref ≈ LAB(166,129,134)). Calibration profiles save/load (`--save-profile`/`--profile`).
**Goal:** obtain the fixed track reference without a pre-recorded run.
- Startup calibration: roll/drive a few seconds on-track, compute + lock the global ref (a "calibrate" action).
- Per-environment config (sim / warehouse / real): thresholds, horizon, fan span (a0/a1), ref.
- Drift handling: manual re-calibrate command; optional slow adaptive update while confidently on-track.
- **DoD:** one-command calibration yields a working ref in a fresh environment.

## Phase 2 — DonkeyCar sim integration (closed loop) ✅ DONE
Drives `generated_track` without leaving the road (0 resets / 1200 steps in the best run). Steering
calibrated in-loop: `ground` weight gain 3.0 beat 1.6 (11→1 reset) and pixel (weaved). **Oscillation
resolved** — root cause was a perceive-twice-per-step recording bug (now perceive once); plain baseline
is smooth (std Δsteer ≈ 0.045), optional deadband/PD-damp available but not needed. Off-track flag
shown live + threshold fixed (grass ~8% vs on-track ~16–18% ⇒ `--offtrack-cov 0.10`).

## Phase 2b — Throttle scheduling (NEXT)
Restore clearance-scaled throttle (currently constant 0.17): faster on straights, slow into curves
(scale by forward clearance / |steer|). Keep survival + smoothness as the bar.

## Phase 3 — Robustness in sim
- Sharp curves: the residual understeer is the far-ray-dilutes-the-turn effect → add **mid-field /
  look-ahead-capped weighting** (weight by ground distance but cap the look-ahead so distant rays
  don't wash out the near road direction). This is the main remaining steering refinement.
- Obstacles (cones): already stop rays; add an avoidance bias (steer away from near stops).
- Generalisation: run multiple sim tracks; verify calibration + thresholds transfer.
- Recovery: 🚧 `raypilot/recovery.py` `RecoveryController` built — detects the coverage collapse EARLY
  (`warn_cov`), slows, then PULSED reverse with steer HELD at last-good (option-1 retrace) until the
  track is re-acquired (`recover_cov`, debounced); `max_reverse` → STUCK safety stop. Wired into
  `drive_gym.py` (`--recovery` on by default, `--warn-cov/--recover-cov/--reverse-throttle/--reverse-steer`).
  Sim-focused for now; the real-car ESC reverse sequence is deferred. **Pending: closed-loop sim test.**
- **DoD:** robust laps across several sim tracks with recovery.

> Note: codebase refactored into a `raypilot/` package (perception `ray_mask`, control `pilot`,
> `recovery`, `donkey_part`) + root entry scripts (`drive_gym`, `drive_physical_raycast`,
> `tune_from_tub`, `render_overlay`); `experiments/` and `legacy/` split out.

## Phase 4 — Real-car port — 🚧 DRIVER BUILT, pending on-Pi field test
**Driver done:** `drive_physical_raycast.py` adapts a proven TFLite physical scaffold (PiCamera
128×120 = our sim res, PCA9685 PWM ch14/ch1, safe shutdown) but runs `RayPilot.perceive()` instead of
a model. Boot = ~3 s straight calibration (creep + sample road → lock colour ref) → autonomous.
Image-only, no model file, runs on a fresh clone; runtime needs only cv2+numpy (matplotlib now lazy).
**Field-verify on the Pi:** (1) FPS of the 80-ray loop — if low, drop `n_rays`/downscale; (2) steering
direction (swap L/R PWM if reversed); (3) `offtrack_cov` for the real surface; (4) gain/`steer_trim`/
asymmetric L-R gain. Workflow target met: **git clone → 3 s straight calibrate → drives.**

**Goal:** run on the physical DonkeyCar (RPi/Jetson + camera + PWM).
- **Compute:** the python ray loop is the main risk on-board. Mitigations: **downscale the image**
  (we proved ray-cast is resolution-invariant — free speed), **fewer rays** (~40), vectorise/numba the march.
  Target ≥15 FPS on-board.
- **Real camera:** different FOV / exposure / white balance → recalibrate thresholds; handle auto-exposure
  (periodic re-cal or illumination normalisation e.g. CLAHE; the global ref may drift with lighting).
- **Real surfaces:** reflections, shadows, texture, non-uniform floor → lean on local-edge + adaptive ref.
- **Control/actuation:** PWM steering/throttle calibration, latency budget, kill-switch, speed cap.
- **Field loop:** tune over real laps; log frames+decisions for offline debugging.
- **DoD:** car completes a real lap on a marked track; graceful off-track stop.

---

## Cross-cutting
- Image-only throughout (no telemetry).
- Logging: persist frame + steer/throttle + off-track for offline debugging (esp. failures).
- Config-driven per environment; safety first (off-track stop, speed caps).

## Top risks → mitigations
1. **Real-time perf on RPi** → downscale (resolution-invariant), fewer rays, vectorise/numba.
2. **Lighting / calibration drift (real)** → adaptive/periodic re-cal, illumination normalisation.
3. **Sharp curves & fan geometry** → free-space-direction control; aim/origin adaptation.
4. **Sim→real gap** → expect a dedicated real recal + tuning pass; FOV/exposure differences.

## Suggested order
✅ Phase 0 → ✅ Phase 1 → ✅ Phase 2 (closed loop, drives the track) → 🚧 Phase 4 driver built
(`drive_physical_raycast.py`), **next = on-Pi field test (FPS, PWM dir, gains/trim)**. Deferred:
Phase 2b throttle scheduling, Phase 3 (sharp-curve mid-field weighting, multi-track, recovery).
