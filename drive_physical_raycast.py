"""Physical DonkeyCar driver using the ray-cast pilot (image-only, classical, no NN/CTE/reward).

Adapted from drive_physical_tflite.py: SAME hardware scaffold (PiCamera 128x120, PCA9685 PWM,
safety shutdown) — the VAE+Actor inference is replaced by RayPilot.perceive(). On boot it runs a
~3 s STRAIGHT calibration on the track (creep straight, sample the road colour, lock the reference),
then drives autonomously. Nothing to train; works on a fresh `git clone`.

    python drive_physical_raycast.py            # calibrate 3s straight, then drive
"""
import sys
import time
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

from donkeycar.parts.actuator import PCA9685
from donkeycar.parts.camera import PiCamera

from ray_pilot import RayPilot
from ray_mask import seed_ref

# --- HARDWARE CONFIG (from your calibrated physical setup) ---
STEERING_LEFT_PWM = 470          # steer = -1  (verify direction on your car; swap with RIGHT if reversed)
STEERING_RIGHT_PWM = 320         # steer = +1
THROTTLE_FORWARD_PWM = 480
THROTTLE_STOPPED_PWM = 400
THROTTLE_REVERSE_PWM = 320

# --- DRIVING KNOBS ---
STEERING_GAIN = 1.0              # final multiplier on the pilot's steer (raise if turns too wide)
THROTTLE_BOOST = 1.0             # multiplier to overcome real-world friction
CONST_THROTTLE = 0.20            # fixed throttle while driving (None -> use the pilot's clearance throttle)
STOP_ON_OFFTRACK = True          # cut throttle when off-track (safety)
CALIB_SECONDS = 3.0              # straight-line calibration duration
CALIB_THROTTLE = 0.18            # gentle forward throttle during calibration

# --- RAY-CAST PILOT PARAMS (sim-tuned defaults; colour ref is set live by calibration) ---
RAY_KW = dict(seed_y=0.85, n_rays=80, a0=8, a1=172, white_margin=90, white_s=60,
              color_thr=40, wl=0.1, horizon=0.35, edge_thr=22, edge_window=4)
CTRL = dict(steer_gain=3.0, base_throttle=0.5, ema=0.4, offtrack_cov=0.10, clear_ref=0.75,
            weight="ground", persp_horizon=0.35, min_gap_frac=0.05,
            steer_trim=0.0, gain_left=None, gain_right=None)
STEER_CENTER_PWM = int((STEERING_LEFT_PWM + STEERING_RIGHT_PWM) / 2)


def to_pwm(val, lo_action, hi_action, lo_pwm, hi_pwm):
    return int((val - lo_action) * (hi_pwm - lo_pwm) / (hi_action - lo_action) + lo_pwm)


class PhysicalRayCar:
    def __init__(self):
        print("\n" + "=" * 50 + f"\nRAY-CAST PILOT | image-only, no model | const_throttle={CONST_THROTTLE}\n" + "=" * 50)
        self.steering = self.throttle = self.camera = None
        try:
            self.pilot = RayPilot(np.array([160, 128, 128], np.float32), 160.0, RAY_KW, **CTRL)
            print("[INFO] Binding PCA9685 (steering ch14, throttle ch1)...")
            self.steering = PCA9685(channel=14, busnum=1)
            self.throttle = PCA9685(channel=1, busnum=1)
            print("[INFO] Initializing PiCamera (128x120)...")
            self.camera = PiCamera(image_w=128, image_h=120)
            time.sleep(2)
            print("[OK] Systems nominal.\n")
        except Exception as e:
            print(f"\n[FATAL] Hardware init failed: {e}")
            self.shutdown(); sys.exit(1)

    def _grab_bgr(self):
        rgb = self.camera.run()                              # PiCamera gives RGB
        return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)

    def calibrate_straight(self, seconds=CALIB_SECONDS):
        """Creep straight on the track for `seconds`, sampling the road colour -> lock the ref."""
        print(f"[CALIB] Hold the car straight on the track. Sampling road for {seconds:.0f}s...")
        self.steering.run(STEER_CENTER_PWM)
        creep_pwm = to_pwm(CALIB_THROTTLE, 0.0, 1.0, THROTTLE_STOPPED_PWM, THROTTLE_FORWARD_PWM)
        self.throttle.run(creep_pwm)
        refs, t0 = [], time.time()
        while time.time() - t0 < seconds:
            refs.append(seed_ref(self._grab_bgr()))          # (LAB, V) of the front seed patch
            time.sleep(0.03)
        self.throttle.run(THROTTLE_STOPPED_PWM)
        if refs:
            self.pilot.ref = np.median(np.array([r for r, _ in refs]), axis=0)
            self.pilot.ref_v = float(np.median([v for _, v in refs]))
            rf = self.pilot.ref
            print(f"[CALIB] locked ref LAB({rf[0]:.0f},{rf[1]:.0f},{rf[2]:.0f}) V{self.pilot.ref_v:.0f} "
                  f"from {len(refs)} frames\n")
        else:
            print("[CALIB] WARNING: no frames captured; using default ref.\n")

    def run(self):
        print("=" * 50 + "\nAUTONOMOUS MODE  (Ctrl+C to stop)\n" + "=" * 50)
        try:
            while True:
                t = time.time()
                r = self.pilot.perceive(self._grab_bgr())
                steer = float(np.clip(r["steer"] * STEERING_GAIN, -1.0, 1.0))
                if STOP_ON_OFFTRACK and r["offtrack"]:
                    thr = 0.0
                else:
                    thr = CONST_THROTTLE if CONST_THROTTLE is not None else r["throttle"]
                    thr = float(np.clip(thr * THROTTLE_BOOST, 0.0, 1.0))
                self.steering.run(to_pwm(steer, -1.0, 1.0, STEERING_LEFT_PWM, STEERING_RIGHT_PWM))
                self.throttle.run(to_pwm(thr, 0.0, 1.0, THROTTLE_STOPPED_PWM, THROTTLE_FORWARD_PWM))
                fps = 1.0 / max(time.time() - t, 1e-3)
                flag = "OFF " if r["offtrack"] else "ON  "
                print(f"FPS {fps:4.1f} | {flag} cov {r['coverage']*100:3.0f}% | steer {steer:+.2f} | thr {thr:.2f}", end="\r")
        except KeyboardInterrupt:
            print("\n[WARN] Manual override.")
        finally:
            self.shutdown()

    def shutdown(self):
        print("\n" + "=" * 50 + "\nSAFE SHUTDOWN\n" + "=" * 50)
        if self.throttle:
            try: self.throttle.run(THROTTLE_STOPPED_PWM)
            except Exception: pass
        if self.steering:
            try: self.steering.run(STEER_CENTER_PWM)
            except Exception: pass
        if self.camera:
            try: self.camera.shutdown()
            except Exception: pass
        print("[DONE] Car secured.")


if __name__ == "__main__":
    car = PhysicalRayCar()
    car.calibrate_straight()
    car.run()
