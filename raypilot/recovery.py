"""Off-track recovery state machine (image-only, no CTE/reward).

Sits BETWEEN the pilot and the actuators, watching the ray-cast `coverage` (drivable-fan area
fraction). Deliberate, debounced sequence so a momentary dip (sharp curve, shadow, crossing a line)
never triggers a reverse:

    DRIVE   coverage healthy                         -> pass the pilot's (steer, throttle) through
    SLOW    coverage < warn_cov (early caution)      -> keep steer, cap throttle, keep watching
    STOP    coverage < off_cov SUSTAINED >= offtrack_secs (1-2 s)  -> full halt for stop_secs
                                                       (settle; an ESC needs neutral before reverse)
    REVERSE back up (pulsed, steer HELD = retrace)   -> until the track is re-acquired
                                                       (coverage >= recover_cov, debounced) -> DRIVE
    STUCK   reversed past max_reverse_secs           -> full stop, wait for the view to return

Timing is in SECONDS (converted to frames via control_hz) so it behaves the same at 20 Hz in sim and
whatever the Pi runs. Telemetry-free: inputs are coverage + the pilot's own steer (both image-derived).
Reverse uses negative throttle (donkey-gym accepts it); the real-car ESC reverse sequence is separate.
"""

DRIVE, SLOW, STOP, REVERSE, STUCK = "DRIVE", "SLOW", "STOP", "REVERSE", "STUCK"


class RecoveryController:
    def __init__(self, warn_cov=0.13, off_cov=0.10, recover_cov=0.15,
                 slow_throttle=0.10, reverse_throttle=-0.30, control_hz=20.0,
                 offtrack_secs=0.5, stop_secs=0.3, max_reverse_secs=3.0, recover_secs=0.3,
                 stuck_secs=1.0, pulse_len=6, pulse_gap=3, reverse_steer_mode="hold"):
        self.warn_cov, self.off_cov, self.recover_cov = warn_cov, off_cov, recover_cov
        self.slow_throttle, self.reverse_throttle = slow_throttle, reverse_throttle
        hz = max(float(control_hz), 1.0)
        # seconds -> frames (so the behaviour is rate-independent)
        self.confirm = max(1, int(round(offtrack_secs * hz)))    # sustained off-track before acting
        self.stop_frames = max(1, int(round(stop_secs * hz)))    # halt duration before reversing
        self.max_reverse = max(1, int(round(max_reverse_secs * hz)))
        self.recover_frames = max(1, int(round(recover_secs * hz)))
        self.stuck_frames = max(1, int(round(stuck_secs * hz)))  # cap on the STUCK halt -> resume forward
        # pulsed reverse: back up `pulse_len` frames, coast `pulse_gap` so perception re-settles
        self.pulse_len, self.pulse_gap = pulse_len, pulse_gap
        self.reverse_steer_mode = reverse_steer_mode             # "hold" (retrace) | "mirror" | "straight"
        self.reset()

    def reset(self):
        self.state = DRIVE
        self.last_good_steer = 0.0     # steer when coverage was last healthy -> held while reversing
        self._low = 0                  # consecutive off-track frames (toward `confirm`)
        self._rec = 0                  # consecutive recovered frames (toward `recover_frames`)
        self._t = 0                    # frames spent in the current STOP / REVERSE phase
        self._pulse_i = 0

    def _reverse_steer(self):
        if self.reverse_steer_mode == "mirror":
            return -self.last_good_steer
        if self.reverse_steer_mode == "straight":
            return 0.0
        return self.last_good_steer    # "hold" = retrace the path we drove in on

    def _reacquired(self, coverage):
        """True once coverage has been back >= recover_cov for recover_frames consecutive frames."""
        if coverage >= self.recover_cov:
            self._rec += 1
            return self._rec >= self.recover_frames
        self._rec = 0
        return False

    def step(self, coverage, steer, throttle):
        """Inputs: ray coverage [0,1], the pilot's commanded steer + proposed throttle.
        Returns (steer, throttle, state). DRIVE passes the inputs through unchanged."""
        if coverage >= self.warn_cov:
            self.last_good_steer = steer       # remember the steer used while the view was healthy

        # ---- DRIVE / SLOW: wait for SUSTAINED off-track before committing to recovery ----
        if self.state in (DRIVE, SLOW):
            self._low = self._low + 1 if coverage < self.off_cov else 0
            if self._low >= self.confirm:      # off-track held >= offtrack_secs -> stop, then reverse
                self.state = STOP; self._t = 0; self._low = 0; self._rec = 0
                return 0.0, 0.0, STOP
            if coverage < self.warn_cov:       # early caution: slow but keep steering, keep watching
                self.state = SLOW
                return steer, min(throttle, self.slow_throttle), SLOW
            self.state = DRIVE
            return steer, throttle, DRIVE

        # ---- STOP: brief full halt to settle; if it recovers here, no reverse needed ----
        if self.state == STOP:
            self._t += 1
            if self._reacquired(coverage):
                self.state = DRIVE; self._rec = 0
                return steer, throttle, DRIVE
            if self._t >= self.stop_frames:
                self.state = REVERSE; self._t = 0; self._pulse_i = 0; self._rec = 0
            return 0.0, 0.0, STOP

        # ---- REVERSE: pulsed back-up with held steer until the track/direction is visible ----
        if self.state == REVERSE:
            if coverage >= self.recover_cov:   # track came back -> SUDDEN STOP, settle while HALTED
                self._rec += 1                 # (do NOT keep reversing past the track during confirm)
                if self._rec >= self.recover_frames:
                    self.state = DRIVE; self._rec = 0
                    return steer, throttle, DRIVE
                return 0.0, 0.0, REVERSE        # halted, not reversing, while confirming
            self._rec = 0                       # lost it again -> back up some more (the back-and-forth)
            self._t += 1
            if self._t >= self.max_reverse:    # reversing isn't working -> brief STUCK, then resume
                self.state = STUCK; self._t = 0
                return 0.0, 0.0, STUCK
            backing = (self._pulse_i % (self.pulse_len + self.pulse_gap)) < self.pulse_len
            self._pulse_i += 1
            return self._reverse_steer(), (self.reverse_throttle if backing else 0.0), REVERSE

        # ---- STUCK: brief halt; never sit stopped for long -> resume forward (moving > frozen) ----
        if self._reacquired(coverage):
            self.state = DRIVE; self._rec = 0
            return steer, throttle, DRIVE
        self._t += 1
        if self._t >= self.stuck_frames:       # give up waiting -> drive forward and try again
            self.state = DRIVE; self._rec = 0
            return steer, throttle, DRIVE
        return 0.0, 0.0, STUCK
