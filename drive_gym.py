"""Phase 2 closed-loop driver: ray-cast pilot in the DonkeyCar gym (gym_donkeycar).

Requires (on YOUR machine, not runnable from the dev sandbox):
    pip install gymnasium gym_donkeycar
    + the DonkeyCar sim binary (DonkeySimMac/Linux/Win) running or path via DONKEY_SIM_PATH.

TELEMETRY-FREE BY DESIGN: the sim's cte / reward / info are never read (the step() helper discards
them). The only sim-side signal used is `done`, and ONLY to reset the crashed car between episodes in
this evaluation harness -- it never reaches perception or control, and the real car (donkey_part.py)
never sees it. The "steps survived" print is a sim-harness convenience derived from that reset signal.

    # 1) make a calibration profile from a recorded tub of the same track (one-off):
    .venv/bin/python ray_pilot.py --img-dir tub_generated_track --white-margin 90 --color-thr 40 \
        --wl 0.1 --horizon 0.35 --edge-thr 22 --weight ground --save-profile calib_sim.json --max-frames 1
    # 2) drive closed loop:
    .venv/bin/python drive_gym.py --profile calib_sim.json --env donkey-generated-track-v0 --steps 2000
"""
import argparse
import os
import time

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Closed-loop ray pilot in donkey-gym")
    p.add_argument("--profile", required=True, help="calibration profile from ray_pilot.py --save-profile")
    p.add_argument("--env", default="donkey-generated-track-v0")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--throttle-scale", type=float, default=1.0)
    p.add_argument("--min-throttle", type=float, default=0.1)
    p.add_argument("--const-throttle", type=float, default=None,
                   help="fixed throttle (steering-calibration mode; car always moves, no deadlock)")
    # steering-calibration overrides (None = keep the profile's value)
    p.add_argument("--steer-gain", type=float, default=None, help="override steering gain")
    p.add_argument("--weight", choices=["pixel", "ground"], default=None, help="override heading weight")
    p.add_argument("--ema", type=float, default=None, help="override steer/throttle smoothing (lower=smoother)")
    p.add_argument("--deadband", type=float, default=None, help="anti-weave deadband on heading error (e.g. 0.1)")
    p.add_argument("--steer-damp", type=float, default=None, help="PD damping term (e.g. 0.4) to smooth overshoot")
    p.add_argument("--live-calib", dest="live_calib", action="store_true", default=True,
                   help="creep forward at start and calibrate the track colour from LIVE frames (default on)")
    p.add_argument("--no-live-calib", dest="live_calib", action="store_false")
    p.add_argument("--warmup-steps", type=int, default=60, help="creep steps for live calibration")
    p.add_argument("--creep-throttle", type=float, default=0.18, help="throttle during the creep/calibrate phase")
    p.add_argument("--stop-offtrack", dest="stop_offtrack", action="store_true", default=True,
                   help="cut throttle when off-track (default); --no-stop-offtrack creeps through instead")
    p.add_argument("--no-stop-offtrack", dest="stop_offtrack", action="store_false")
    p.add_argument("--sim-path",
                   default=os.environ.get("DONKEY_SIM_PATH",
                       "/Users/harshwadhawe/Downloads/DonkeySimMac/donkey_sim.app/Contents/MacOS/donkey_sim"),
                   help="path to the sim binary, or 'remote' to attach to an already-running sim")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9091)
    p.add_argument("--record", default=None, help="optional mp4 of the camera + overlay")
    return p.parse_args()


def main():
    a = parse_args()
    try:
        import gymnasium as gym                      # donkeycar 5.2 / gym_donkeycar 1.3+ use gymnasium
    except ImportError:
        try:
            import gym
        except ImportError:
            print("Missing deps. In your donkey env:  pip install gymnasium gym_donkeycar")
            return
    try:
        import gym_donkeycar  # noqa: F401  (registers the donkey envs)
    except ImportError:
        print("Missing gym_donkeycar in this env.")
        return

    import cv2
    from donkey_part import RayPilotPart
    from ray_pilot import draw

    part = RayPilotPart(a.profile, throttle_scale=a.throttle_scale, min_throttle=a.min_throttle,
                        stop_on_offtrack=a.stop_offtrack, const_throttle=a.const_throttle)
    if a.steer_gain is not None:                              # live calibration overrides
        part.pilot.steer_gain = a.steer_gain
    if a.weight is not None:
        part.pilot.weight = a.weight
    if a.ema is not None:
        part.pilot.ema = a.ema
    if a.deadband is not None:
        part.pilot.steer_deadband = a.deadband
    if a.steer_damp is not None:
        part.pilot.steer_damp = a.steer_damp
    print(f"steering: gain={part.pilot.steer_gain} weight={part.pilot.weight} ema={part.pilot.ema} "
          f"deadband={part.pilot.steer_deadband} damp={part.pilot.steer_damp}")
    conf = {"host": a.host, "port": a.port, "car_name": "ray-pilot"}
    if a.sim_path != "remote":                          # else: attach to an already-running sim
        conf["exe_path"] = a.sim_path
    env = gym.make(a.env, conf=conf)

    def reset(e):                                        # gym (obs) vs gymnasium (obs, info)
        out = e.reset()
        return out[0] if isinstance(out, tuple) else out

    def step(e, action):                                # -> (obs, done). reward + info are
        res = e.step(action)                            # DISCARDED on purpose: no telemetry (cte/
        if len(res) == 5:                               # reward) is ever read. `done` is used only
            obs, _, term, trunc, _ = res                # to reset the car in the sim harness, never
            return obs, bool(term or trunc)             # for perception/control (the real car never
        obs, _, done, _ = res                           # sees it).
        return obs, bool(done)

    def get_img(obs):                                   # obs may be (img, info) or img
        arr = np.asarray(obs[0] if isinstance(obs, (tuple, list)) else obs)
        return arr if arr.ndim == 3 and arr.shape[2] == 3 else None

    obs = reset(env)

    # ---- Phase 1 live calibration: creep forward, sample the LIVE road colour, lock the ref ----
    if a.live_calib:
        from ray_mask import seed_ref
        refs = []
        for i in range(a.warmup_steps):
            obs, done = step(env, np.array([0.0, a.creep_throttle], np.float32))
            img = get_img(obs)
            if img is not None and i >= a.warmup_steps // 2:   # second half: car is on clean road
                refs.append(seed_ref(cv2.cvtColor(img, cv2.COLOR_RGB2BGR)))
            if done:
                obs = reset(env)
        if refs:
            part.pilot.ref = np.median(np.array([r for r, _ in refs]), axis=0)
            part.pilot.ref_v = float(np.median([v for _, v in refs]))
            rf = part.pilot.ref
            print(f"live calibrated ref LAB({rf[0]:.0f},{rf[1]:.0f},{rf[2]:.0f}) "
                  f"V{part.pilot.ref_v:.0f} (from {len(refs)} live frames)")

    writer = None
    survived, episodes = 0, 0
    t0, steers = time.time(), []
    for step_i in range(a.steps):
        angle, throttle = part.run(obs)                 # perceives obs ONCE (stored on the part)
        steers.append(angle)
        if a.record and getattr(part, "last_r", None) is not None:
            frame = draw(part.last_bgr, part.last_r)     # draw the SAME perception used for control
            if writer is None:
                H, W = frame.shape[:2]
                writer = cv2.VideoWriter(a.record, cv2.VideoWriter_fourcc(*"mp4v"), 20, (W, H))
            writer.write(frame)
        obs, done = step(env, np.array([angle, throttle], dtype=np.float32))
        survived += 1
        if done:                                        # left track / timed out
            episodes += 1
            print(f"  episode end @ step {step_i} (survived {survived} steps)")
            obs = reset(env); survived = 0
    env.close()
    if writer is not None:
        writer.release(); print(f"wrote {a.record}")
    fps = a.steps / (time.time() - t0)
    sm = float(np.std(np.diff(steers))) if len(steers) > 2 else 0.0   # weave: std of frame-to-frame Δsteer
    print(f"done. {a.steps} steps, {episodes} resets, control loop ~{fps:.0f} Hz "
          f"| steer smoothness (std Δsteer) {sm:.3f}  mean|steer| {np.mean(np.abs(steers)):.2f}")


if __name__ == "__main__":
    main()
