"""
Closed-loop motion primitives for the pyhut Duckiebot -- app-side control policy.

Why this lives here and not in pyhut
------------------------------------
pyhut's drivetrain is deliberately *dumb and safe*: you set a target wheel speed
and a 0.3 s watchdog zeros the motors if you stop feeding it. It has no notion of
"turn 90 degrees". Closing a loop around measured heading to hit a target angle
is *policy*, and policy belongs in the application, built on pyhut's ABCs so any
robot that implements them can be driven by the same primitive.

What ``rotate`` does
--------------------
It turns the robot in place at a commanded angular *rate* and stops at a target
*angle*, using MEASURED heading -- never open-loop "spin for N seconds", which
just trusts the motors and accumulates error. The measurement is the fused
heading from :class:`~duckiebot_lab.motion.heading.HeadingTracker` (gyro-primary,
mag if Stage 2b cleared it, encoders as a magnitude cross-check).

Control structure (cascade)
---------------------------
* **Outer / position loop.** From the signed heading error it computes a target
  angular rate, saturated at the requested ``rate_dps`` and tapered down as the
  error shrinks so the robot decelerates into the target instead of overshooting.
* **Inner / rate loop.** A PI loop drives the *measured* angular rate to that
  target by setting a base wheel magnitude (feed-forward + PI). Because it closes
  on measured rate, it absorbs the asymmetry between the two drive paths (left =
  all-PCA channels; right = GPIO direction + PCA PWM): if one side is weaker for
  a given command, the measured rate falls short and the loop pushes harder. A
  per-side trim gives it a feed-forward head start.

Both loops run fast (default 50 Hz), and every tick calls
``drivetrain.set_target`` so the watchdog never trips mid-turn.

Direction sign
--------------
Which wheel pairing (+left/-right vs -left/+right) makes the *measured* heading
increase depends on how the IMU is mounted and how the motors are wired -- it is
a hardware fact, not something to hard-code. On first use the primitive measures
it (:meth:`calibrate_direction`) with a brief nudge and caches the sign, so the
loop is correct on any build. Pass ``dir_sign`` to skip the probe.

Slow by default
---------------
Defaults favour a slow spin: less wheel scrub/slip, more heading samples per
degree, less measurement smear. Push ``rate_dps`` up only if you need speed.

Standard library + pyhut only; Python 3.6-safe.
"""

import time
from collections import namedtuple

from .heading import HeadingTracker, estimate_gyro_bias, ang_diff, wrap180


# ---------------------------------------------------------------------------
# Configuration and result types
# ---------------------------------------------------------------------------
class RotateConfig(object):
    """Tunables for :meth:`MotionPrimitives.rotate`. Plain mutable object (no
    dataclass -- py3.6). Defaults are chosen slow-and-safe; override per call."""

    def __init__(self):
        # Loop timing.
        self.loop_hz = 50.0              # control rate; << 1/0.3 s watchdog
        self.settle_ticks = 4            # consecutive in-tolerance ticks to stop

        # Tolerances / stop conditions.
        self.tol_deg = 1.5               # |error| under this counts as arrived
        self.settle_rate_dps = 8.0       # and measured |rate| must be under this
        self.timeout_margin = 3.0        # timeout = margin * ideal_time + 1.5 s

        # Outer (position) loop.
        self.kp_pos = 3.0                # rate_ref = kp_pos * error (deg/s / deg)

        # Inner (rate) loop -- base wheel magnitude from rate error.
        self.kff = 1.0 / 180.0           # feed-forward: base per deg/s of ref
        self.kp_rate = 1.0 / 300.0       # P on (rate_ref - measured_rate)
        self.ki_rate = 1.0 / 600.0       # I on the same, per second
        self.i_clamp = 0.5               # anti-windup clamp on the integral term
        self.max_base = 0.85             # ceiling on |wheel magnitude|
        self.min_move_base = 0.18        # stiction floor while a move is commanded

        # Per-side asymmetry feed-forward (multiplicative). The inner loop still
        # closes the gap; these just reduce how much it has to.
        self.left_trim = 1.0
        self.right_trim = 1.0

        # Direction probe (used when dir_sign is unknown).
        self.probe_base = 0.35
        self.probe_time_s = 0.30

        # Pre-spin residual-bias estimate.
        self.bias_estimate_s = 0.4
        self.bias_still_std_max = 3.0    # deg/s; above this, stillness is suspect


# Outcome of a rotate call. Everything a caller/telemetry/lidar-demo needs.
RotateResult = namedtuple("RotateResult", [
    "requested_deg",     # commanded angle (signed)
    "achieved_deg",      # fused heading change actually measured (signed)
    "error_deg",         # requested - achieved
    "reached",           # bool: stopped within tolerance (vs timeout)
    "duration_s",        # wall time of the closed-loop phase
    "n_samples",         # heading updates consumed
    "mag_fused",         # bool: was the mag fused live?
    "dir_sign",          # +1/-1 sign used
    "enc_magnitude_deg", # independent encoder magnitude estimate
    "enc_disagreement",  # |gyro heading| - |encoder magnitude|
    "gyro_bias_dps",     # residual bias removed before the spin
    "reason",            # short human string
])


# ---------------------------------------------------------------------------
# The primitives
# ---------------------------------------------------------------------------
class MotionPrimitives(object):
    """Reusable closed-loop motions over a pyhut ``Robot``.

    ``rotate`` is implemented now. ``drive_straight`` is intentionally left as a
    seam: it reuses the same Sampler-fed loop and the gyro-primary
    ``HeadingTracker`` (to hold a line) -- see the stub at the bottom.
    """

    def __init__(self, robot, mag_policy=None,
                 track_width_m=0.10, wheel_diameter_m=0.067, ticks_per_rev=None,
                 dir_sign=None):
        """
        robot          a pyhut Robot (needs .drivetrain and .sampler).
        mag_policy     a MagPolicy from duckiebot_lab.motion.verdict, or None.
                       If it says fuse=True, the mag is fused live during turns.
        track_width_m,
        wheel_diameter_m,
        ticks_per_rev  geometry for the encoder magnitude cross-check. Defaults
                       are Duckiebot-typical; MEASURE these on your bot for the
                       cross-check to mean anything (they do not affect the
                       gyro-primary steering, only the sanity signal).
        dir_sign       +1/-1 to skip the direction probe; None to auto-detect.
        """
        self.robot = robot
        self.drivetrain = robot.drivetrain
        self.sampler = robot.sampler
        if self.sampler is None:
            raise RuntimeError(
                "rotate needs a sampler for feedback, but robot.sampler is None "
                "(no sensors fitted?)")

        self.mag_policy = mag_policy

        self.track_width_m = track_width_m
        self.wheel_diameter_m = wheel_diameter_m
        if ticks_per_rev is None:
            enc = robot.left_encoder or robot.right_encoder
            ticks_per_rev = getattr(enc, "ticks_per_rev", 135) if enc else 135
        self.ticks_per_rev = ticks_per_rev

        self.dir_sign = dir_sign         # cached after first probe

    # -- sampler helpers --
    def _ensure_sampling(self):
        """Make sure the sampler is running and return a first Sample."""
        self.sampler.start()             # idempotent
        s = self.sampler.latest()
        deadline = time.monotonic() + 1.0
        while s is None and time.monotonic() < deadline:
            time.sleep(0.005)
            s = self.sampler.latest()
        if s is None:
            raise RuntimeError("sampler produced no Sample within 1 s")
        return s

    def _fresh_sample(self, last_t):
        """Poll latest() until a Sample newer than last_t appears (bounded)."""
        deadline = time.monotonic() + 0.25
        s = self.sampler.latest()
        while (s is None or s.t == last_t) and time.monotonic() < deadline:
            time.sleep(0.002)
            s = self.sampler.latest()
        return s

    # -- direction sign auto-detect --
    def calibrate_direction(self, cfg=None):
        """Nudge briefly and measure whether commanding (+left,-right) makes the
        measured heading increase. Caches and returns ``dir_sign`` (+1/-1)."""
        cfg = cfg or RotateConfig()
        self._ensure_sampling()
        bias, _, _ = estimate_gyro_bias(self.sampler, cfg.bias_estimate_s)

        # Probe in two equal, opposite pulses so the net rotation cancels and the
        # robot ends near where it started -- the sign is read from the first
        # pulse; the second just undoes it. (Reading the sign is the point; not
        # leaving an offset before the caller's first real turn is the courtesy.)
        base = cfg.probe_base

        def _pulse(left, right, measure):
            rates = []
            end = time.monotonic() + cfg.probe_time_s
            last_t = None
            while time.monotonic() < end:
                s = self.sampler.latest()
                if measure and s is not None and s.gyro is not None and s.t != last_t:
                    rates.append(s.gyro[2] - bias)
                    last_t = s.t
                self.drivetrain.set_target(left, right)   # keep watchdog fed
                time.sleep(1.0 / cfg.loop_hz)
            return rates

        rates = _pulse(+base, -base, measure=True)     # read the sign here
        _pulse(-base, +base, measure=False)            # undo the rotation
        self.drivetrain.stop()

        mean_rate = sum(rates) / len(rates) if rates else 0.0
        # If (+left,-right) produced a positive measured rate, that pairing raises
        # the heading, so dir_sign = +1; otherwise it must be flipped.
        self.dir_sign = 1 if mean_rate >= 0.0 else -1
        # Let the bot settle before the caller starts a real move.
        time.sleep(0.15)
        return self.dir_sign

    # -- the rotate primitive --
    def rotate(self, angle_deg, rate_dps=45.0, cfg=None):
        """Turn ``angle_deg`` in place (sign = direction) at ~``rate_dps``.

        Positive ``angle_deg`` means "measured heading increases"; the wheel
        pairing that achieves that is resolved from the cached/auto-detected
        direction sign, so the caller reasons in heading, not in wheels.

        Returns a :class:`RotateResult`.
        """
        cfg = cfg or RotateConfig()
        rate_dps = abs(rate_dps)
        first = self._ensure_sampling()

        # 1) Residual gyro-bias re-zero (temperature drift since calibration).
        self.drivetrain.stop()
        bias, nbias, bias_std = estimate_gyro_bias(self.sampler, cfg.bias_estimate_s)
        bias_suspect = bias_std > cfg.bias_still_std_max

        # 2) Resolve direction sign if we don't have it yet.
        if self.dir_sign is None:
            self.calibrate_direction(cfg)
            # bias may be slightly stale after the nudge; re-estimate cheaply.
            bias, nbias, bias_std = estimate_gyro_bias(self.sampler, cfg.bias_estimate_s)

        # 3) Decide mag fusion from the Stage 2b policy at THIS operating point.
        #    Operating duty ~ the base magnitude the rate loop will hover at for
        #    the commanded rate (feed-forward estimate), floored by stiction.
        op_duty = max(cfg.min_move_base, min(cfg.max_base, cfg.kff * rate_dps))
        mag_fused = False
        mag_reason = "gyro-only"
        if self.mag_policy is not None:
            # The policy was decided for a nominal duty; re-check against op_duty
            # only tightens MARGINAL, so honour fuse=True but respect max_safe.
            if self.mag_policy.fuse and (
                    self.mag_policy.max_safe_duty is None
                    or op_duty <= self.mag_policy.max_safe_duty + 1e-9):
                mag_fused = True
                mag_reason = self.mag_policy.reason

        # 4) Set up the fused heading tracker, anchored to "now".
        tracker = HeadingTracker(
            gyro_axis=2, gyro_bias_dps=bias,
            use_mag=mag_fused, mag_alpha=0.98,
            track_width_m=self.track_width_m,
            wheel_diameter_m=self.wheel_diameter_m,
            ticks_per_rev=self.ticks_per_rev)
        anchor = self.sampler.latest() or first
        tracker.reset(anchor)

        target = float(angle_deg)                 # relative target (heading==0 now)
        period = 1.0 / cfg.loop_hz
        ideal_time = abs(angle_deg) / rate_dps if rate_dps > 0 else 0.0
        timeout = cfg.timeout_margin * ideal_time + 1.5

        integ = 0.0
        in_tol = 0
        n = 0
        t0 = time.monotonic()

        # Consume the FULL sample stream for heading, not just whatever the
        # control tick happens to poll: an integral that skips intervals
        # under-/over-counts. The estimator runs at sensor rate off the
        # subscription; the controller reads the freshest estimate at its own
        # (possibly slower) cadence. This is also how it should look on hardware.
        q, unsub = self.sampler.subscribe_queue(maxsize=256)
        try:
            while True:
                now = time.monotonic()
                if now - t0 > timeout:
                    reached = False
                    reason = "timeout after %.2fs" % (now - t0)
                    break

                # Drain every sample that arrived since the last tick, feeding
                # each into the heading integral with the current commanded sign.
                got = False
                commanded_sign = 1 if ang_diff(target, tracker.heading) >= 0 else -1
                deadline = now + 0.25
                while True:
                    try:
                        s = q.get_nowait()
                    except Exception:
                        s = None
                    if s is None:
                        if got or time.monotonic() > deadline:
                            break
                        time.sleep(0.002)
                        continue
                    tracker.update(s, commanded_sign)
                    n += 1
                    got = True
                if not got:
                    # No feedback at all this window: keep the watchdog fed.
                    self.drivetrain.stop()
                    continue

                error = ang_diff(target, tracker.heading)   # signed, (-180,180]

                # --- stop test: within tolerance AND actually slowing/stopped ---
                if (abs(error) <= cfg.tol_deg
                        and abs(tracker.gyro_rate) <= cfg.settle_rate_dps):
                    in_tol += 1
                    if in_tol >= cfg.settle_ticks:
                        reached = True
                        reason = "reached target within %.1f deg" % cfg.tol_deg
                        break
                else:
                    in_tol = 0

                # --- outer loop: heading error -> desired angular rate ---
                rate_ref = cfg.kp_pos * error
                if rate_ref > rate_dps:
                    rate_ref = rate_dps
                elif rate_ref < -rate_dps:
                    rate_ref = -rate_dps

                # --- inner loop: measured rate -> base wheel magnitude (PI+FF) ---
                rate_err = rate_ref - tracker.gyro_rate
                integ += rate_err * period
                if integ > cfg.i_clamp:
                    integ = cfg.i_clamp
                elif integ < -cfg.i_clamp:
                    integ = -cfg.i_clamp
                base = (cfg.kff * rate_ref
                        + cfg.kp_rate * rate_err
                        + cfg.ki_rate * integ)

                # Saturate.
                if base > cfg.max_base:
                    base = cfg.max_base
                elif base < -cfg.max_base:
                    base = -cfg.max_base

                # Never brake by reversing while still short of the target. When
                # the rate loop wants to drive *against* the remaining error
                # (measured rate overshot the small commanded rate near the goal)
                # the stiction floor below would slam that reversal to full floor
                # magnitude and set up a limit cycle around the target. Coasting
                # instead eases us in; with real inertia this is also the natural
                # way to decelerate.
                if base * error < 0.0:
                    base = 0.0

                # Inside the tolerance band, stop driving and coast so the rate
                # can fall far enough to satisfy the settle test (the floor can't
                # creep arbitrarily slowly, so we arrive by cutting power, not by
                # nulling the last fraction of a degree under power).
                if abs(error) <= cfg.tol_deg:
                    base = 0.0
                    integ = 0.0          # drop the integral so it can't wind up
                # Otherwise overcome stiction when we do need to move to target.
                elif 0.0 < abs(base) < cfg.min_move_base:
                    base = cfg.min_move_base if base > 0 else -cfg.min_move_base

                # --- map to wheels: +base == +heading via dir_sign; trim L/R ---
                u = base * self.dir_sign
                left = +u * cfg.left_trim
                right = -u * cfg.right_trim
                self.drivetrain.set_target(left, right)

                # pace the loop
                dt_sleep = period - (time.monotonic() - now)
                if dt_sleep > 0:
                    time.sleep(dt_sleep)
        finally:
            unsub()

        # --- stop and settle, then read the final achieved heading ---
        self.drivetrain.stop()
        time.sleep(0.15)
        s = self.sampler.latest()
        if s is not None:
            tracker.update(s, 1 if ang_diff(target, tracker.heading) >= 0 else -1)

        achieved = tracker.heading
        result = RotateResult(
            requested_deg=float(angle_deg),
            achieved_deg=achieved,
            error_deg=ang_diff(target, achieved),
            reached=reached,
            duration_s=time.monotonic() - t0,
            n_samples=n,
            mag_fused=mag_fused,
            dir_sign=self.dir_sign,
            enc_magnitude_deg=tracker.enc_magnitude,
            enc_disagreement=tracker.encoder_disagreement(),
            gyro_bias_dps=bias,
            reason=(reason + ("; BIAS SUSPECT (std %.1f dps)" % bias_std
                              if bias_suspect else "")
                    + "; " + mag_reason),
        )
        return result

    # -- future seam: drive_straight reuses the same machinery --
    def drive_straight(self, distance_m, speed=0.4, cfg=None):
        """Not implemented yet.

        Planned: hold heading with the same gyro-primary HeadingTracker while
        commanding equal wheel speeds, correcting the L/R asymmetry as a small
        differential; measure distance from encoder magnitude (sign known from
        the commanded direction). Left as a seam so rotate could land first.
        """
        raise NotImplementedError(
            "drive_straight is a planned primitive; rotate is implemented")
