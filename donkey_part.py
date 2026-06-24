"""DonkeyCar Part wrapping the ray-cast pilot (image -> steering, throttle). Image-only.

Drop into a DonkeyCar vehicle assembly:

    from donkey_part import RayPilotPart
    V.add(RayPilotPart("calib_sim.json"), inputs=["cam/image_array"], outputs=["angle", "throttle"])

The Part is duck-typed (no donkeycar import), so it also runs standalone / in the gym loop. DonkeyCar
hands camera frames as RGB numpy; ray_mask works in BGR, so we convert. Off-track -> throttle 0.
"""
import cv2
import numpy as np

from ray_pilot import RayPilot


class RayPilotPart:
    def __init__(self, profile, throttle_scale=1.0, min_throttle=0.0, stop_on_offtrack=True,
                 const_throttle=None):
        self.pilot = RayPilot.from_profile(profile)
        self.throttle_scale = throttle_scale
        self.min_throttle = min_throttle
        self.stop_on_offtrack = stop_on_offtrack
        self.const_throttle = const_throttle      # fixed throttle (steering-calibration mode)
        self.last = (0.0, 0.0)

    def run(self, img_arr):
        """img_arr: HxWx3 RGB uint8 (DonkeyCar camera). Returns (angle, throttle) in [-1,1], [0,1]."""
        if isinstance(img_arr, (tuple, list)):       # gymnasium reset() gives (obs, info)
            img_arr = img_arr[0]
        if img_arr is None:
            return self.last
        arr = np.asarray(img_arr)
        if arr.ndim != 3 or arr.shape[2] != 3:       # not a usable frame yet
            return self.last
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        r = self.pilot.perceive(bgr)
        angle = float(np.clip(r["steer"], -1.0, 1.0))
        if self.const_throttle is not None:                       # steering-calibration mode: always move
            throttle = float(self.const_throttle)
        elif r["offtrack"] and self.stop_on_offtrack:
            throttle = 0.0
        else:
            throttle = float(np.clip(max(self.min_throttle, r["throttle"] * self.throttle_scale), 0.0, 1.0))
        self.last = (angle, throttle)
        return angle, throttle

    def shutdown(self):
        pass
