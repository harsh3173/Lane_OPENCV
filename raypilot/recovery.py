"""Off-track recovery state machine (image-only, no CTE/reward).

Sits BETWEEN the pilot and the actuators. It watches the ray-cast `coverage` (drivable-fan area
fraction) — the same signal the off-track flag uses — and runs:

    DRIVE   coverage healthy                  -> pass the pilot's (steer, throttle) straight through
    SLOW    coverage < warn_cov (early sign)  -> keep the pilot's steer, cut throttle (caution)
    REVERSE coverage < off_cov (confirmed)    -> PULSED reverse, steer HELD at last-good value
                                                 (option 1: retrace the entry path) until the track
                                                 is re-acquired (coverage back >= recover_cov)
    STUCK   reversed past max_reverse frames  -> full stop, hold, wait for coverage to return

Re-acquisition needs coverage back up for `recover_frames` consecutive frames (not a 1-frame flicker)
before resuming forward -- "see the track lines, THEN continue". All transitions are debounced.

Telemetry-free: the only inputs are coverage + the pilot's own steer command (both image-derived).
The sim's cte/reward/done never enter here. Reverse uses negative throttle (donkey-gym accepts it);
the real-car ESC reverse sequence is a separate concern, deferred (sim focus for now).
"""

DRIVE, SLOW, REVERSE, STUCK = "DRIVE", "SLOW", "REVERSE", "STUCK"


class RecoveryController:
    def __init__(self, warn_cov=0.13, off_cov=0.10, recover_cov=0.15,
                 slow_throttle=0.10, reverse_throttle=-0.30,
                 pulse_len=6, pulse_gap=3, max_reverse=120,
                 confirm_frames=4, recover_frames=4, reverse_steer_mode="hold"):
        # thresholds (warn > off so we slow BEFORE we're off; recover > warn for hysteresis)
        self.warn_cov, self.off_cov, self.recover_cov = warn_cov, off_cov, recover_cov
        self.slow_throttle, self.reverse_throttle = slow_throttle, reverse_throttle
        # pulsed reverse: back up `pulse_len` frames, coast `pulse_gap` frames so perception re-settles
        self.pulse_len, self.pulse_gap = pulse_len, pulse_gap
        self.max_reverse = max_reverse                 # safety cap -> STUCK (never reverse forever)
        self.confirm_frames, self.recover_frames = confirm_frames, recover_frames
        self.reverse_steer_mode = reverse_steer_mode   # "hold" (option 1 retrace) | "mirror" | "straight"
        self.reset()

    def reset(self):
        self.state = DRIVE
        self.last_good_steer = 0.0     # steer when coverage was last healthy -> held while reversing
        self._low = self._rec = 0      # debounce counters (entering reverse / re-acquiring)
        self._rev_frames = 0           # total frames spent reversing this episode (safety cap)
        self._pulse_i = 0              # position within a reverse pulse cycle

    def _reverse_steer(self):
        if self.reverse_steer_mode == "mirror":
            return -self.last_good_steer
        if self.reverse_steer_mode == "straight":
            return 0.0
        return self.last_good_steer    # "hold" = retrace the path we drove in on (option 1)

    def step(self, coverage, steer, throttle):
        """Inputs: ray coverage [0,1], the pilot's commanded steer, its proposed throttle.
        Returns (steer, throttle, state). In DRIVE the inputs pass through unchanged."""
        # remember the steer used while we still had a healthy view -> what we retrace on
        if coverage >= self.warn_cov:
            self.last_good_steer = steer

        if self.state == DRIVE:
            if coverage < self.off_cov:
                self._low += 1
                if self._low >= self.confirm_frames:
                    self.state = REVERSE; self._low = 0
                    self._rev_frames = self._pulse_i = 0
                    return self._reverse_steer(), 0.0, REVERSE   # neutral frame before reverse
            else:
                self._low = 0
                if coverage < self.warn_cov:
                    self.state = SLOW
            return steer, throttle, self.state

        if self.state == SLOW:
            if coverage >= self.warn_cov:
                self.state = DRIVE
                return steer, throttle, DRIVE
            if coverage < self.off_cov:
                self._low += 1
                if self._low >= self.confirm_frames:
                    self.state = REVERSE; self._low = 0
                    self._rev_frames = self._pulse_i = 0
                    return self._reverse_steer(), 0.0, REVERSE
            else:
                self._low = 0
            return steer, min(throttle, self.slow_throttle), SLOW   # caution: cap throttle, keep steer

        if self.state == REVERSE:
            # re-acquired? need coverage back, sustained, before resuming forward
            if coverage >= self.recover_cov:
                self._rec += 1
                if self._rec >= self.recover_frames:
                    self.state = DRIVE; self._rec = 0
                    return steer, throttle, DRIVE
            else:
                self._rec = 0
            self._rev_frames += 1
            if self._rev_frames >= self.max_reverse:
                self.state = STUCK
                return 0.0, 0.0, STUCK
            # pulsed reverse: hold-steer back-up, then a short coast so perception can update
            cycle = self.pulse_len + self.pulse_gap
            backing = (self._pulse_i % cycle) < self.pulse_len
            self._pulse_i += 1
            thr = self.reverse_throttle if backing else 0.0
            return self._reverse_steer(), thr, REVERSE

        # STUCK: full stop, hold steer centered; auto-clear only if the view comes back on its own
        if coverage >= self.recover_cov:
            self._rec += 1
            if self._rec >= self.recover_frames:
                self.state = DRIVE; self._rec = 0
                return steer, throttle, DRIVE
        else:
            self._rec = 0
        return 0.0, 0.0, STUCK
