# Ray-Pilot Roadmap — offline masks → live DonkeyCar (sim → real)

Porting `ray_mask.py` (radial free-space ray-cast) from offline tub analysis into a live, image-only
autonomous driving controller. No CTE/telemetry is ever used (see memory: no-cte-reward).

## Baseline (where we are)
- `ray_mask.py`: rays fan from bottom-centre, stop at white lines / colour edges; **global colour
  calibration** (fixed track ref → off-track collapses) + **local-edge stop** + **horizon cap**.
- Validated offline on recorded tubs: sim road, old-car mat, warehouse floor (per-domain params).
- Output today: drivable mask + ray endpoints + off-track flag. Image-only.
- Known limit: the fixed-origin symmetric fan can't fully wrap a *sharp* curve in the **mask** — but a
  free-space-**direction** controller can still steer through it (control needs a heading, not a perfect mask).

---

## Phase 0 — Core perception+control module (offline)
**Goal:** turn the mask into a (steer, throttle) command in a fast, reusable, dependency-light module.
- Extract `ray_perception(image, cfg) -> {mask, endpoints, offtrack, steer, throttle}` (drop matplotlib/CLI/video).
- **Steering law** (pick by experiment): free-space *balance* (left vs right free area) or longest-free-ray
  heading → normalised steer [-1,1]. Prefer free-space heading over mask-centroid — it handles curves.
- **Throttle:** base value, scaled by forward free distance; → 0 (or reverse/stop) when off-track.
- **Temporal smoothing:** EMA on steer (we already proved this kills jitter).
- **Validate offline:** overlay the predicted steer arrow on the existing tub videos; arrow should track the road.
- **Perf:** profile the ray loop; target <30–50 ms/frame on dev. Lever: fewer rays, vectorise the march.
- **DoD:** steer arrow follows the road on sim+warehouse tub videos; meets FPS target.

## Phase 1 — Live calibration
**Goal:** obtain the fixed track reference without a pre-recorded run.
- Startup calibration: roll/drive a few seconds on-track, compute + lock the global ref (a "calibrate" action).
- Per-environment config (sim / warehouse / real): thresholds, horizon, fan span (a0/a1), ref.
- Drift handling: manual re-calibrate command; optional slow adaptive update while confidently on-track.
- **DoD:** one-command calibration yields a working ref in a fresh environment.

## Phase 2 — DonkeyCar sim integration (closed loop)
**Goal:** drive autonomously in donkey-gym.
- Implement a DonkeyCar **Part** `RayPilot` (camera img → angle/throttle), or a donkey-gym control loop.
- Wire off-track → safe behaviour (cut throttle / stop / simple recovery).
- Closed-loop tune steering gain, look-ahead (seed-y / horizon), throttle in the sim.
- Metrics: laps completed, % time off-track, steering smoothness, avg speed.
- **DoD:** completes laps on `generated_track` in donkey-gym without leaving the road.

## Phase 3 — Robustness in sim
- Sharp curves: confirm the free-space controller handles them despite the mask limit; if not, adapt the
  aim (steer toward longest-free-ray) or shift the ray origin.
- Obstacles (cones): already stop rays; add an avoidance bias (steer away from near stops).
- Generalisation: run multiple sim tracks; verify calibration + thresholds transfer.
- Recovery: from off-track / near-wall states.
- **DoD:** robust laps across several sim tracks with recovery.

## Phase 4 — Real-car port
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
Phase 0 (control law + speed) → Phase 2 (sim closed loop) → Phase 1 hardens calibration as you go →
Phase 3 (sim robustness) → Phase 4 (real). Phase 0's steering law is the single highest-value next step.
