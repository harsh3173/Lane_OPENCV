"""Phase 2 closed-loop driver: ray-cast pilot in the DonkeyCar gym (gym_donkeycar).

Requires (on YOUR machine, not runnable from the dev sandbox):
    pip install gymnasium gym_donkeycar
    + the DonkeyCar sim binary (DonkeySimMac/Linux/Win) running or path via DONKEY_SIM_PATH.

TELEMETRY-FREE BY DESIGN: the sim's cte / reward / info are never read (the step() helper discards
them). The only sim-side signal used is `done`, and ONLY to reset the crashed car between episodes in
this evaluation harness -- it never reaches perception or control, and the real car (donkey_part.py)
never sees it. The "steps survived" print is a sim-harness convenience derived from that reset signal.

    # 1) make a calibration profile from a recorded tub of the same track (one-off):
    .venv/bin/python render_overlay.py --img-dir tub_generated_track --white-margin 90 --color-thr 40 \
        --wl 0.1 --horizon 0.35 --edge-thr 22 --weight ground --save-profile calib_sim.json --max-frames 1
    # 2) drive closed loop:
    .venv/bin/python drive_gym.py --profile calib_sim.json --env donkey-generated-track-v0 --steps 2000
"""
import argparse
import collections
import os
import signal
import subprocess
import time

import numpy as np


class SimTimeout(Exception):
    """Raised when a sim step/reset blocks longer than --sim-timeout (the sim died or hung)."""


def _on_alarm(signum, frame):
    raise SimTimeout()


def cleanup_stale(verbose=True):
    """Kill leftover donkey_sim binaries and other hung drive_gym.py clients (never ourselves), so a
    fresh run gets a clean sim + free port instead of attaching to / fighting a zombie."""
    me = os.getpid()
    subprocess.run(["pkill", "-f", "donkey_sim"], capture_output=True)
    try:
        out = subprocess.run(["pgrep", "-f", "drive_gym.py"], capture_output=True, text=True).stdout
        killed = [pid for pid in out.split() if pid and int(pid) != me]
        for pid in killed:
            try:
                os.kill(int(pid), signal.SIGKILL)
            except Exception:
                pass
        if verbose and killed:
            print(f"[cleanup] killed stale drive_gym pid(s): {', '.join(killed)}")
    except Exception:
        pass


def parse_args():
    p = argparse.ArgumentParser(description="Closed-loop ray pilot in donkey-gym")
    p.add_argument("--profile", required=True, help="calibration profile from render_overlay.py --save-profile")
    p.add_argument("--steer-mode", choices=["ray", "flow"], default="ray",
                   help="ray = base free-space-heading pilot (default); flow = goal flow-field steering")
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
    p.add_argument("--cov-ema", type=float, default=0.5, help="flow: coverage EMA for off-track (lower=smoother/laggier; ~0.5 = short)")
    p.add_argument("--deadband", type=float, default=None, help="anti-weave deadband on heading error (e.g. 0.1)")
    p.add_argument("--steer-damp", type=float, default=None, help="PD damping term (e.g. 0.4) to smooth overshoot")
    p.add_argument("--offtrack-cov", type=float, default=None, help="coverage below this = off-track (e.g. 0.10)")
    p.add_argument("--steer-trim", type=float, default=None, help="constant steer offset to cancel center/camera bias")
    p.add_argument("--gain-left", type=float, default=None, help="steering gain for LEFT turns (asymmetry)")
    p.add_argument("--gain-right", type=float, default=None, help="steering gain for RIGHT turns (asymmetry)")
    p.add_argument("--shadow-pass", dest="shadow_pass", action="store_true", default=None,
                   help="shadow-robust rays: pass through dark+neutral patches (shadows/trees), stop on chroma/bright")
    p.add_argument("--no-shadow-pass", dest="shadow_pass", action="store_false")
    p.add_argument("--green-stop", dest="green_stop", action="store_true", default=None,
                   help="stop rays on green+dark grass (off-track); on by default in ray_mask")
    p.add_argument("--no-green-stop", dest="green_stop", action="store_false")
    # ray penetration overrides (None = use the profile's value) -- experiment with deeper rays
    p.add_argument("--color-thr", type=float, default=None, help="override ray colour-stop threshold (higher = rays penetrate more)")
    p.add_argument("--horizon", type=float, default=None, help="override horizon cap y-fraction (0 = no cap, rays climb to the top)")
    p.add_argument("--edge-thr", type=float, default=None, help="override local-edge stop threshold (higher = penetrate more)")
    p.add_argument("--live-calib", dest="live_calib", action="store_true", default=True,
                   help="creep forward at start and calibrate the track colour from LIVE frames (default on)")
    p.add_argument("--no-live-calib", dest="live_calib", action="store_false")
    p.add_argument("--warmup-steps", type=int, default=60, help="creep steps for live calibration")
    p.add_argument("--creep-throttle", type=float, default=0.18, help="throttle during the creep/calibrate phase")
    p.add_argument("--stop-offtrack", dest="stop_offtrack", action="store_true", default=True,
                   help="cut throttle when off-track (default); --no-stop-offtrack creeps through instead")
    p.add_argument("--no-stop-offtrack", dest="stop_offtrack", action="store_false")
    # --- off-track recovery (detect early -> slow -> hold-steer reverse until re-acquired) ---
    p.add_argument("--recovery", dest="recovery", action="store_true", default=True,
                   help="enable the reverse-recovery state machine (default on)")
    p.add_argument("--no-recovery", dest="recovery", action="store_false")
    p.add_argument("--warn-cov", type=float, default=0.13, help="coverage below this -> SLOW (early caution)")
    p.add_argument("--recover-cov", type=float, default=0.15, help="coverage above this (sustained) -> resume forward")
    p.add_argument("--reverse-throttle", type=float, default=-0.15, help="throttle while backing up (negative; gentler = stops quicker)")
    p.add_argument("--reverse-pulse", type=int, default=3, help="frames of reverse per step (smaller = more minimal back-up steps)")
    p.add_argument("--reverse-gap", type=int, default=2, help="coast frames between reverse pulses (re-check coverage)")
    p.add_argument("--reverse-steer", choices=["align", "hold", "mirror", "straight"], default="align",
                   help="steer while reversing: align (rotate parallel to track), hold (retrace), mirror, straight")
    p.add_argument("--align-gain", type=float, default=1.0, help="align mode: how hard to rotate toward parallel")
    p.add_argument("--align-thr", type=float, default=0.30, help="align mode: |steer| below this = aligned -> resume")
    # time-based recovery sequencing (seconds; rate-independent via --control-hz)
    p.add_argument("--offtrack-secs", type=float, default=0.5, help="off-track must persist this long before STOP+reverse")
    p.add_argument("--stop-secs", type=float, default=0.3, help="full-halt duration before reversing")
    p.add_argument("--max-reverse-secs", type=float, default=3.0, help="max time reversing before STUCK (safety cap)")
    p.add_argument("--stuck-secs", type=float, default=1.0, help="max time fully stopped in STUCK before resuming forward")
    p.add_argument("--control-hz", type=float, default=20.0, help="control-loop rate, to convert the *-secs into frames")
    p.add_argument("--sim-path",
                   default=os.environ.get("DONKEY_SIM_PATH",
                       "/Users/harshwadhawe/Downloads/DonkeySimMac/donkey_sim.app/Contents/MacOS/donkey_sim"),
                   help="path to the sim binary, or 'remote' to attach to an already-running sim")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9091)
    p.add_argument("--record", default=None, help="optional mp4 of the camera + overlay")
    p.add_argument("--dump-frames", default=None, help="dir: save RAW camera frames (no overlay) here for offline threshold tuning")
    p.add_argument("--dump-stride", type=int, default=2, help="save every Nth frame when --dump-frames")
    p.add_argument("--episode-pre-secs", type=float, default=2.5,
                   help="with --dump-frames: also save this many seconds of frames before each episode termination to episode_ends/ (the off-track + recovery window)")
    # scripted disturbance to TEST recovery (the pilot is too good to go off on its own)
    p.add_argument("--test-maneuver", action="store_true",
                   help="periodically force a hard turn off-track (alternating L/R) to exercise recovery")
    p.add_argument("--disturb-start", type=float, default=35.0, help="seconds to drive cleanly before the FIRST forced turn (~2 laps)")
    p.add_argument("--disturb-every", type=float, default=20.0, help="seconds between forced off-track turns (~once a lap+)")
    p.add_argument("--disturb-dur", type=float, default=1.0, help="seconds of forced hard turn (drives off the track)")
    p.add_argument("--disturb-steer", type=float, default=1.0, help="steer magnitude of the forced turn")
    p.add_argument("--no-cleanup", dest="cleanup", action="store_false", default=True,
                   help="skip killing stale donkey_sim / drive_gym processes before launch")
    p.add_argument("--sim-timeout", type=int, default=15,
                   help="abort cleanly if a sim step blocks longer than this many seconds (0 = no watchdog)")
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
    from raypilot.recovery import RecoveryController

    # --- build the steering agent (uniform interface: run/last_r/last_bgr; calib_target has ref/ref_v) ---
    if a.steer_mode == "flow":
        from raypilot.pilot import RayPilot
        from raypilot.flow_field import FlowPart, draw as draw_fn
        base = RayPilot.from_profile(a.profile)              # profile gives ref / ref_v / ray_kw
        if a.shadow_pass is not None:
            base.ray_kw["shadow_pass"] = a.shadow_pass
        if a.green_stop is not None:
            base.ray_kw["green_stop"] = a.green_stop
        for k, val in (("color_thr", a.color_thr), ("horizon", a.horizon), ("edge_thr", a.edge_thr)):
            if val is not None:
                base.ray_kw[k] = val
        agent = FlowPart(base.ref, base.ref_v, base.ray_kw,
                         const_throttle=(a.const_throttle if a.const_throttle is not None else 0.17),
                         stop_on_offtrack=a.stop_offtrack,
                         steer_gain=(a.steer_gain if a.steer_gain is not None else 3.4),  # validated ray-heading gain
                         ema=(a.ema if a.ema is not None else 0.4), cov_ema=a.cov_ema,
                         offtrack_cov=(a.offtrack_cov if a.offtrack_cov is not None else 0.10))
        calib_target, off_cov = agent.flow, agent.flow.offtrack_cov
        print(f"steering: FLOW field | gain={agent.flow.steer_gain} ema={agent.flow.ema} "
              f"offtrack_cov={off_cov} shadow_pass={base.ray_kw.get('shadow_pass')}")
    else:
        from raypilot.donkey_part import RayPilotPart
        from raypilot.pilot import draw as draw_fn
        agent = RayPilotPart(a.profile, throttle_scale=a.throttle_scale, min_throttle=a.min_throttle,
                             stop_on_offtrack=a.stop_offtrack, const_throttle=a.const_throttle)
        p = agent.pilot
        if a.steer_gain is not None: p.steer_gain = a.steer_gain      # live calibration overrides
        if a.weight is not None: p.weight = a.weight
        if a.ema is not None: p.ema = a.ema
        if a.deadband is not None: p.steer_deadband = a.deadband
        if a.steer_damp is not None: p.steer_damp = a.steer_damp
        if a.offtrack_cov is not None: p.offtrack_cov = a.offtrack_cov
        if a.steer_trim is not None: p.steer_trim = a.steer_trim
        if a.gain_left is not None: p.gain_left = a.gain_left
        if a.gain_right is not None: p.gain_right = a.gain_right
        if a.shadow_pass is not None: p.ray_kw["shadow_pass"] = a.shadow_pass
        if a.green_stop is not None: p.ray_kw["green_stop"] = a.green_stop
        for k, val in (("color_thr", a.color_thr), ("horizon", a.horizon), ("edge_thr", a.edge_thr)):
            if val is not None: p.ray_kw[k] = val
        calib_target, off_cov = p, p.offtrack_cov
        print(f"steering: RAY base | gain={p.steer_gain} weight={p.weight} ema={p.ema} "
              f"shadow_pass={p.ray_kw.get('shadow_pass')} trim={p.steer_trim} gainL={p.gain_left} gainR={p.gain_right}")
    if a.cleanup:                                       # clear zombie sim/clients so we get a clean port
        cleanup_stale()
    if a.sim_timeout > 0:                               # watchdog: a dead/hung sim can't freeze us
        signal.signal(signal.SIGALRM, _on_alarm)

    conf = {"host": a.host, "port": a.port, "car_name": "ray-pilot"}
    if a.sim_path != "remote":                          # else: attach to an already-running sim
        conf["exe_path"] = a.sim_path
    env = gym.make(a.env, conf=conf)

    def reset(e):                                        # gym (obs) vs gymnasium (obs, info)
        signal.alarm(a.sim_timeout if a.sim_timeout > 0 else 0)
        out = e.reset()
        signal.alarm(0)
        return out[0] if isinstance(out, tuple) else out

    def step(e, action):                                # -> (obs, done). reward + info are
        signal.alarm(a.sim_timeout if a.sim_timeout > 0 else 0)
        res = e.step(action)                            # DISCARDED on purpose: no telemetry (cte/
        signal.alarm(0)
        if len(res) == 5:                               # reward) is ever read. `done` is used only
            obs, _, term, trunc, _ = res                # to reset the car in the sim harness, never
            return obs, bool(term or trunc)             # for perception/control (the real car never
        obs, _, done, _ = res                           # sees it).
        return obs, bool(done)

    def get_img(obs):                                   # obs may be (img, info) or img
        arr = np.asarray(obs[0] if isinstance(obs, (tuple, list)) else obs)
        return arr if arr.ndim == 3 and arr.shape[2] == 3 else None

    writer = recov = None
    survived, episodes, dumped, ep_saved = 0, 0, 0, 0
    t0, steers, rev_steps = time.time(), [], 0
    rec_active, rec_start, rec_durs = False, 0, []        # recovery-event durations (STOP/REVERSE/STUCK)
    try:
        obs = reset(env)

        # ---- Phase 1 live calibration: creep forward, sample the LIVE road colour, lock the ref ----
        if a.live_calib:
            from raypilot.ray_mask import seed_ref
            refs = []
            for i in range(a.warmup_steps):
                obs, done = step(env, np.array([0.0, a.creep_throttle], np.float32))
                img = get_img(obs)
                if img is not None and i >= a.warmup_steps // 2:   # second half: car is on clean road
                    refs.append(seed_ref(cv2.cvtColor(img, cv2.COLOR_RGB2BGR)))
                if done:
                    obs = reset(env)
            if refs:
                calib_target.ref = np.median(np.array([r for r, _ in refs]), axis=0)
                calib_target.ref_v = float(np.median([v for _, v in refs]))
                rf = calib_target.ref
                print(f"live calibrated ref LAB({rf[0]:.0f},{rf[1]:.0f},{rf[2]:.0f}) "
                      f"V{calib_target.ref_v:.0f} (from {len(refs)} live frames)")

        if a.recovery:
            recov = RecoveryController(warn_cov=a.warn_cov, off_cov=off_cov, recover_cov=a.recover_cov,
                                       reverse_throttle=a.reverse_throttle, control_hz=a.control_hz,
                                       offtrack_secs=a.offtrack_secs, stop_secs=a.stop_secs,
                                       max_reverse_secs=a.max_reverse_secs, stuck_secs=a.stuck_secs,
                                       pulse_len=a.reverse_pulse, pulse_gap=a.reverse_gap,
                                       reverse_steer_mode=a.reverse_steer,
                                       align_gain=a.align_gain, align_thr=a.align_thr)
            print(f"recovery ON: confirm {a.offtrack_secs}s off<{off_cov} -> STOP {a.stop_secs}s -> reverse "
                  f"(thr {a.reverse_throttle}, steer {a.reverse_steer}, max {a.max_reverse_secs}s) -> "
                  f"recover>{a.recover_cov} | @{a.control_hz:.0f}Hz")

        if a.dump_frames:
            os.makedirs(a.dump_frames, exist_ok=True)
        # rolling buffer of the most recent frames -> flushed to episode_ends/ on each termination,
        # so we can inspect exactly what the camera saw (and what recovery did) right before going off
        ep_buf = collections.deque(maxlen=max(1, int(a.episode_pre_secs * a.control_hz)))

        period = max(1, int(a.disturb_every * a.control_hz))
        dur = max(1, int(a.disturb_dur * a.control_hz))
        dstart = int(a.disturb_start * a.control_hz)        # drive clean for ~2 laps before disturbing
        for step_i in range(a.steps):
            angle, throttle = agent.run(obs)                # perceives obs ONCE (stored on the agent)
            if a.dump_frames and step_i % a.dump_stride == 0 and getattr(agent, "last_bgr", None) is not None:
                cv2.imwrite(os.path.join(a.dump_frames, f"{step_i:05d}.jpg"), agent.last_bgr)  # RAW, no overlay
                dumped += 1
            # --- scripted disturbance: drive clean for disturb_start, then FORCE a hard turn off-track
            # at the end of each cycle (alternating L/R = the mirror), then release -> recovery catches it
            sd = step_i - dstart
            disturbing = a.test_maneuver and sd >= 0 and (sd % period) < dur   # turn at the start of each cycle
            forced_info = None
            if disturbing:
                side = -1.0 if (sd // period) % 2 == 0 else 1.0      # left first, then its mirror (right)
                angle = side * a.disturb_steer
                throttle = a.const_throttle if a.const_throttle is not None else 0.20
                forced_info = ("LEFT" if side < 0 else "RIGHT", (sd % period) / a.control_hz, a.disturb_dur)
            if getattr(agent, "last_r", None) is not None:
                agent.last_r["forced"] = forced_info          # for the overlay flag (None when not forced)
            rstate = "FORCED" if disturbing else "DRIVE"
            if recov is not None and not disturbing and getattr(agent, "last_r", None) is not None:
                angle, throttle, rstate = recov.step(agent.last_r["coverage"], angle, throttle)
                if rstate in ("REVERSE", "STUCK"):
                    rev_steps += 1
                is_rec = rstate in ("STOP", "REVERSE", "STUCK")   # a recovery EVENT (not SLOW caution)
                if is_rec and not rec_active:
                    rec_active, rec_start = True, step_i
                elif not is_rec and rec_active:
                    rec_active = False; rec_durs.append(step_i - rec_start)
            if a.dump_frames and getattr(agent, "last_bgr", None) is not None:   # every step (stride 1)
                cov = agent.last_r["coverage"] if getattr(agent, "last_r", None) else -1
                ep_buf.append((step_i, rstate, cov, agent.last_bgr.copy()))
            steers.append(angle)
            if a.record and getattr(agent, "last_r", None) is not None:
                agent.last_r["recovery"] = rstate            # surface state on the overlay (ray draw uses it)
                frame = draw_fn(agent.last_bgr, agent.last_r)   # draw the SAME perception used for control
                if writer is None:
                    H, W = frame.shape[:2]
                    writer = cv2.VideoWriter(a.record, cv2.VideoWriter_fourcc(*"mp4v"), 20, (W, H))
                writer.write(frame)
            obs, done = step(env, np.array([angle, throttle], dtype=np.float32))
            survived += 1
            if done:                                        # left track / timed out
                episodes += 1
                print(f"  episode end @ step {step_i} (survived {survived} steps)")
                if a.dump_frames and ep_buf:                 # set aside the off-track + recovery window
                    d = os.path.join(a.dump_frames, "episode_ends", f"ep{episodes:02d}_step{step_i:05d}")
                    os.makedirs(d, exist_ok=True)
                    for si, rs, cov, fr in ep_buf:           # filename encodes step, recovery state, coverage
                        cv2.imwrite(os.path.join(d, f"{si:05d}_{rs}_cov{int(cov*100):03d}.jpg"), fr)
                    ep_saved += len(ep_buf)
                    print(f"    -> set aside {len(ep_buf)} pre-termination frames -> {d}")
                    ep_buf.clear()
                obs = reset(env); survived = 0
                if recov is not None:
                    recov.reset()                            # fresh state machine for the new episode
        if rec_active:                                    # close an episode still open at the end
            rec_durs.append(a.steps - rec_start)
        fps = a.steps / (time.time() - t0)
        hz = fps if fps > 1 else a.control_hz             # seconds-per-step from the real loop rate
        sm = float(np.std(np.diff(steers))) if len(steers) > 2 else 0.0   # weave: std of Δsteer
        print(f"done. {a.steps} steps, {episodes} resets, control loop ~{fps:.0f} Hz "
              f"| steer smoothness (std Δsteer) {sm:.3f}  mean|steer| {np.mean(np.abs(steers)):.2f}")
        if recov is not None:
            if rec_durs:
                ds = sorted(d / hz for d in rec_durs)
                print(f"recovery: {len(rec_durs)} event(s), {rev_steps} active steps | "
                      f"duration mean {np.mean(ds):.1f}s  max {ds[-1]:.1f}s  total {sum(ds):.1f}s")
            else:
                print("recovery: not triggered (car stayed on track)")
    except KeyboardInterrupt:
        print("\n[interrupted] Ctrl+C — shutting the sim down cleanly.")
    except SimTimeout:
        print(f"\n[watchdog] sim stalled >{a.sim_timeout}s (died or hung) — aborting cleanly.")
    finally:                                            # ALWAYS release the sim + port (no zombies)
        signal.alarm(0)
        try:
            env.close()
        except Exception:
            pass
        if writer is not None:
            writer.release(); print(f"wrote {a.record}")
        if a.dump_frames:
            print(f"dumped {dumped} raw frames -> {a.dump_frames}/ | "
                  f"{ep_saved} frames set aside across {episodes} episode_ends/")


if __name__ == "__main__":
    main()
