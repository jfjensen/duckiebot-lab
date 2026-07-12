"""
Heading estimation for the app-side motion primitives.

This module turns pyhut's timestamped ``Sample`` stream into a running heading
estimate, in the exact fusion order the hardware dictates:

  1. **Primary: integrated, bias-corrected gyro-z.** ``Sample.gyro[2]`` is deg/s.
     When the robot is built with a pyhut ``Calibration`` the sampler is wired
     over the *calibrated* IMU view, so that value already has the gyro bias
     removed. We additionally re-zero any *residual* bias right before a motion
     (see :func:`estimate_gyro_bias`), because a gyro's zero-rate output drifts
     with temperature and the freshest possible offset is the one measured
     seconds before the spin. Integrating the corrected rate over the measured
     ``dt`` (from consecutive ``Sample.t`` values, not wall-clock) gives heading.

  2. **Conditional: magnetometer, complementary-filtered.** The mag is an
     *absolute* heading reference that stops the gyro integral drifting over a
     long turn -- but only if Stage 2b's EMI verdict says it is usable while the
     motors run. It is fused as a slow correction (weight ``1 - alpha``, with
     ``alpha`` near 1 so the gyro dominates short-term). We anchor the mag to the
     gyro frame at reset and fuse its *unwrapped delta*, so an unknown compass
     offset between the two frames never matters. Whether this runs at all is
     decided upstream (see :mod:`duckiebot_lab.motion.verdict`) and passed in as
     ``use_mag``.

  3. **Cross-check only: encoders as magnitude.** The Duckiebot's Hall encoders
     count edges up-only; they cannot sense direction. So they never feed the
     primary heading. Instead we turn tick deltas into an *independent magnitude*
     of rotation and apply the commanded sign, purely to catch a stalled wheel,
     gross slip, or a dead gyro -- never as the steering signal.

The class is deliberately reusable: a future ``drive_straight`` primitive wants
the same gyro-primary heading tracker to hold a line, so the fusion lives here
rather than inside ``rotate``.

Standard library only, Python 3.6-safe (no dataclasses, no f-strings required).
"""

import math
import time


# Degrees of gyro-integrated rotation to accumulate before we trust the mag's
# rotation sense (used to reconcile the compass sign against the gyro).
_MAG_RECONCILE_DEG = 10.0


# ---------------------------------------------------------------------------
# Angle helpers -- all degrees.
# ---------------------------------------------------------------------------
def wrap360(a):
    """Fold an angle into [0, 360)."""
    return a % 360.0


def wrap180(a):
    """Fold an angle difference into (-180, 180]."""
    a = (a + 180.0) % 360.0 - 180.0
    # (-180, 180]: nudge the -180 boundary to +180 for symmetry.
    return a + 360.0 if a <= -180.0 else a


def ang_diff(a, b):
    """Signed smallest difference a - b, in (-180, 180]."""
    return wrap180(a - b)


def _atan2_heading_deg(field):
    """Tilt-naive compass heading in [0, 360) from a calibrated (x, y, z) field.

    Mirrors pyhut's ``Magnetometer.heading_deg`` so a mag heading computed here
    from ``Sample.mag`` matches what the driver would report.
    """
    if field is None:
        return None
    mx, my = field[0], field[1]
    return (math.degrees(math.atan2(my, mx)) + 360.0) % 360.0


# ---------------------------------------------------------------------------
# Pre-motion residual gyro-bias estimate
# ---------------------------------------------------------------------------
def estimate_gyro_bias(sampler, duration_s=0.4, axis=2, settle_s=0.05):
    """Average gyro-``axis`` over ``duration_s`` while the robot is held still.

    Returns ``(bias_dps, samples, std_dps)``. ``bias_dps`` is the residual
    zero-rate offset to subtract during integration; ``std_dps`` lets a caller
    sanity-check stillness (a large std means the bot was not actually stationary
    -- e.g. still coasting -- and the estimate should not be trusted).

    Uses the Sampler for feedback like everything else; it does not read the IMU
    directly, so it never contends with the sampler for the I2C bus.
    """
    deadline = time.monotonic() + max(0.0, settle_s)
    while time.monotonic() < deadline:
        time.sleep(0.005)

    vals = []
    last_t = None
    end = time.monotonic() + duration_s
    while time.monotonic() < end:
        s = sampler.latest()
        if s is not None and s.gyro is not None and s.t != last_t:
            vals.append(s.gyro[axis])
            last_t = s.t
        time.sleep(0.005)

    if not vals:
        return 0.0, 0, 0.0
    mean = sum(vals) / len(vals)
    if len(vals) > 1:
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    return mean, len(vals), std


# ---------------------------------------------------------------------------
# The heading tracker
# ---------------------------------------------------------------------------
class HeadingTracker(object):
    """Running heading from the pyhut Sample stream.

    Heading starts at 0 at :meth:`reset` and accumulates *relative* to that
    zero, which is exactly what an in-place rotation wants (turn N degrees from
    wherever you are). Absolute compass heading is available via the fused mag if
    a caller needs it, but the primitive steers on this relative estimate.
    """

    def __init__(self,
                 gyro_axis=2,
                 gyro_bias_dps=0.0,
                 use_mag=False,
                 mag_alpha=0.98,
                 track_width_m=0.10,
                 wheel_diameter_m=0.067,
                 ticks_per_rev=135):
        # --- gyro (primary) ---
        self.gyro_axis = gyro_axis
        self.gyro_bias_dps = gyro_bias_dps
        # --- mag (conditional) ---
        self.use_mag = use_mag
        self.mag_alpha = mag_alpha
        # --- encoder (cross-check only) ---
        self.track_width_m = track_width_m
        self.wheel_circumference_m = math.pi * wheel_diameter_m
        self.ticks_per_rev = ticks_per_rev

        self.reset()

    # -- lifecycle --
    def reset(self, sample=None):
        """Zero the estimate. If a first ``sample`` is given, anchor to it."""
        self.heading = 0.0            # fused, relative to reset, degrees
        self.gyro_heading = 0.0       # pure gyro integral, for diagnostics
        self.gyro_rate = 0.0          # latest bias-corrected gyro rate, deg/s
        self._last_t = None
        # mag anchoring
        self._mag_ref = None          # mag heading at reset (absolute, deg)
        self._mag_unwrapped = 0.0     # continuous mag delta from ref, deg
        self._mag_prev = None
        self.mag_heading = None       # latest absolute mag heading, deg
        # mag-sign reconciliation: the compass may increase or decrease as the
        # gyro-z heading increases, depending on axis convention. We don't assume
        # -- we measure it against the known-good gyro once enough rotation has
        # accumulated. 0 = not yet reconciled (gyro-only until then).
        self._mag_sign = 0
        self.mag_reconciled = False
        # encoder anchoring
        self._enc_l0 = None
        self._enc_r0 = None
        self.enc_magnitude = 0.0      # |rotation| from encoders, deg
        self.enc_signed = 0.0         # magnitude with commanded sign applied
        if sample is not None:
            self._anchor(sample)

    def _anchor(self, sample):
        self._last_t = sample.t
        if sample.mag is not None:
            h = _atan2_heading_deg(sample.mag)
            self._mag_ref = h
            self._mag_prev = h
            self.mag_heading = h
        if sample.enc_left is not None:
            self._enc_l0 = sample.enc_left
        if sample.enc_right is not None:
            self._enc_r0 = sample.enc_right

    # -- update --
    def update(self, sample, commanded_sign=0):
        """Fold one Sample into the estimate. ``commanded_sign`` (+1/-1/0) is the
        sign of the rotation currently commanded, applied to the (direction-
        blind) encoder magnitude. Returns the current fused heading (deg)."""
        if sample is None:
            return self.heading
        if self._last_t is None:
            self._anchor(sample)
            return self.heading

        dt = sample.t - self._last_t
        if dt <= 0.0:
            return self.heading          # stale/duplicate sample; skip
        self._last_t = sample.t

        # 1) primary: integrate bias-corrected gyro-z
        if sample.gyro is not None:
            self.gyro_rate = sample.gyro[self.gyro_axis] - self.gyro_bias_dps
            self.gyro_heading += self.gyro_rate * dt
            self.heading += self.gyro_rate * dt

        # 2) conditional: complementary mag correction toward absolute heading
        if self.use_mag and sample.mag is not None and self._mag_ref is not None:
            h = _atan2_heading_deg(sample.mag)
            self.mag_heading = h
            # unwrap mag into a continuous delta from its reset reference
            step = ang_diff(h, self._mag_prev)
            self._mag_unwrapped += step
            self._mag_prev = h

            # Reconcile the mag's rotation sense against the gyro before trusting
            # it. Until we've turned enough for both to have a clear direction,
            # stay gyro-only; then lock the sign that makes them agree.
            if self._mag_sign == 0:
                if (abs(self.gyro_heading) >= _MAG_RECONCILE_DEG
                        and abs(self._mag_unwrapped) >= 0.5 * _MAG_RECONCILE_DEG):
                    same = (self._mag_unwrapped * self.gyro_heading) >= 0.0
                    self._mag_sign = 1 if same else -1
                    self.mag_reconciled = True
            else:
                # mag's estimate of our *relative* heading, in the gyro's frame
                mag_rel = self._mag_sign * self._mag_unwrapped
                # blend: gyro-dominant, mag nudges out slow drift
                err = ang_diff(mag_rel, self.heading)
                self.heading += (1.0 - self.mag_alpha) * err

        # 3) cross-check only: encoder magnitude with commanded sign
        if (sample.enc_left is not None and sample.enc_right is not None
                and self._enc_l0 is not None and self._enc_r0 is not None):
            dl = abs(sample.enc_left - self._enc_l0)
            dr = abs(sample.enc_right - self._enc_r0)
            # in-place spin: both wheels travel opposite by ~equal arc; robot
            # yaw magnitude ~= (arc_left + arc_right) / track_width.
            arc_l = (dl / float(self.ticks_per_rev)) * self.wheel_circumference_m
            arc_r = (dr / float(self.ticks_per_rev)) * self.wheel_circumference_m
            yaw_rad = (arc_l + arc_r) / self.track_width_m
            self.enc_magnitude = math.degrees(yaw_rad)
            self.enc_signed = self.enc_magnitude * (1 if commanded_sign >= 0 else -1)

        return self.heading

    # -- diagnostics --
    def encoder_disagreement(self):
        """|gyro heading| vs |encoder magnitude|, in degrees.

        A large value late in a turn means the two independent sources disagree
        on how far we've turned -- a wheel is slipping/stalled, or the gyro is
        bad. Callers can surface it or abort.
        """
        return abs(abs(self.gyro_heading) - self.enc_magnitude)
