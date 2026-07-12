"""
A headless simulation robot for exercising the motion primitives without
hardware. It implements pyhut's ABCs (``Robot``, ``Drivetrain``, ``Sampler``)
with a simple but physically-shaped in-place rotation plant, and injects the
exact nasties the primitive must survive:

  * **Drive asymmetry** -- left and right gains differ, so equal commanded
    magnitudes give unequal wheel speeds (mirrors all-PCA left vs GPIO+PCA right).
  * **Deadband / stiction** -- commands below a floor produce no motion.
  * **Gyro bias + noise** -- a nonzero zero-rate offset and per-sample noise, so
    the pre-spin re-zero and the integration have something to correct.
  * **Mounting sign** -- ``plant_sign`` flips the relationship between the
    commanded wheel pairing and the sign of measured gyro-z, so the direction
    auto-detect is actually tested.
  * **Magnetometer** -- a rotating Earth field plus an optional EMI offset, so
    mag fusion (when enabled) has a real absolute reference.

Real threads, real time, real ``time.monotonic`` timestamps: the primitive runs
against it exactly as it would against hardware. Kept intentionally small; it is
a test/`--simulate` aid, not a fidelity simulator.
"""

import math
import threading
import time

from pyhut.interfaces import (
    Sample, Drivetrain, Sampler, WheelEncoder, Robot,
)


class _SimState(object):
    """Shared plant state, advanced by the sampler thread from the drive target."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.left = 0.0
        self.right = 0.0
        self.true_heading = 0.0      # deg, ground truth
        self.enc_left = 0            # up-only tick counts
        self.enc_right = 0
        self._enc_l_acc = 0.0        # fractional tick accumulators
        self._enc_r_acc = 0.0
        # Optional simulated caster/wheel bind -- see SimConfig.bind_deg.
        self._bind_crossed = False
        self._bound = False
        self._bind_free_progress = 0.0

    def set_target(self, left, right):
        with self.lock:
            self.left = max(-1.0, min(1.0, left))
            self.right = max(-1.0, min(1.0, right))

    def _wheel_speed(self, cmd, gain):
        """Commanded magnitude -> effective signed wheel speed with deadband."""
        c = self.cfg
        if abs(cmd) < c.deadband:
            return 0.0
        return cmd * gain

    def advance(self, dt):
        """Integrate the plant by dt seconds. Returns (gyro_z_dps, mag_field)."""
        c = self.cfg
        with self.lock:
            left, right = self.left, self.right
        ls = self._wheel_speed(left, c.left_gain)    # "m/s-ish" units
        rs = self._wheel_speed(right, c.right_gain)

        # In-place yaw rate ~ (right - left) scaled; plant_sign is the mounting.
        yaw_rate_dps = c.plant_sign * (rs - ls) * c.yaw_scale   # deg/s (commanded)
        dtheta_deg = yaw_rate_dps * dt

        # Optional simulated caster/wheel bind: once the true heading first
        # reaches bind_deg, the BODY stops responding to further drive -- the
        # wheels still turn (ticks keep accruing from the commanded motion
        # below) but yaw doesn't -- until bind_free_reverse_deg of *reverse*
        # commanded rotation has accumulated, which frees it again. Mirrors a
        # caught caster: the encoders see motion the gyro doesn't.
        effective_dtheta = dtheta_deg
        if c.bind_deg is not None:
            if not self._bind_crossed and (
                    (self.true_heading < c.bind_deg <= self.true_heading + dtheta_deg)
                    or (self.true_heading > c.bind_deg >= self.true_heading + dtheta_deg)):
                self._bind_crossed = True
                self._bound = True
            if self._bound:
                if dtheta_deg < 0.0:
                    self._bind_free_progress += -dtheta_deg
                    if self._bind_free_progress >= c.bind_free_reverse_deg:
                        self._bound = False
                effective_dtheta = 0.0
        self.true_heading += effective_dtheta

        # Encoders (up-only): driven from the COMMANDED rotation, not the
        # possibly bind-blocked true one -- a caught caster still lets the
        # driven wheels themselves turn. For a pure in-place spin each wheel
        # travels an arc of |dtheta_rad| * (track_width/2).
        arc = abs(math.radians(dtheta_deg)) * (c.track_width_m / 2.0)   # metres
        ticks = (arc / c.wheel_circumference_m) * c.ticks_per_rev
        self._enc_l_acc += ticks
        self._enc_r_acc += ticks
        add_l = int(self._enc_l_acc); self._enc_l_acc -= add_l
        add_r = int(self._enc_r_acc); self._enc_r_acc -= add_r
        self.enc_left += add_l
        self.enc_right += add_r

        # Gyro-z reports the TRUE (possibly bind-blocked) rate + bias + noise.
        true_rate_dps = effective_dtheta / dt if dt > 0.0 else 0.0
        noise = c.rng_gyro()
        gyro_z = true_rate_dps + c.gyro_bias_dps + noise

        # Mag: Earth field rotated by true heading + optional EMI (sensor-frame).
        th = math.radians(self.true_heading)
        bx = c.earth_uT * math.cos(-th)
        by = c.earth_uT * math.sin(-th)
        # EMI grows with commanded effort (worst near full duty).
        effort = 0.5 * (abs(left) + abs(right))
        ex, ey = c.emi_uT * effort, 0.0
        mag = (bx + ex, by + ey, 0.0)
        return gyro_z, mag


class SimConfig(object):
    def __init__(self):
        self.left_gain = 1.00
        self.right_gain = 0.85      # right path weaker -> asymmetry to compensate
        self.deadband = 0.12        # commands below this don't move the wheels
        self.yaw_scale = 90.0       # deg/s at unit wheel-speed differential
        self.plant_sign = 1         # +1 or -1: mounting of gyro vs wheel pairing
        self.gyro_bias_dps = 1.3    # residual zero-rate offset
        self.gyro_noise_dps = 0.4   # 1-sigma per-sample noise
        self.earth_uT = 30.0
        self.emi_uT = 0.0           # set >0 to disturb the mag with motor effort
        # Geometry, kept consistent with MotionPrimitives' defaults so the
        # encoder magnitude cross-check reflects the true turn (a fair test).
        self.track_width_m = 0.10
        self.wheel_circumference_m = math.pi * 0.067
        self.ticks_per_rev = 135
        # Simulated caster/wheel bind (see _SimState.advance); None disables it.
        self.bind_deg = None
        self.bind_free_reverse_deg = 3.0
        import random
        self._r = random.Random(12345)

    def rng_gyro(self):
        return self._r.gauss(0.0, self.gyro_noise_dps)


# --- pyhut ABC implementations ------------------------------------------------
class SimDrivetrain(Drivetrain):
    def __init__(self, state):
        self._state = state

    def set_target(self, left, right):
        self._state.set_target(left, right)

    def stop(self):
        self._state.set_target(0.0, 0.0)

    def close(self):
        self.stop()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class SimEncoder(WheelEncoder):
    ticks_per_rev = 135

    def __init__(self, state, side):
        self._state = state
        self._side = side

    @property
    def ticks(self):
        return self._state.enc_left if self._side == "left" else self._state.enc_right

    def reset(self):
        pass

    def close(self):
        pass


class SimSampler(Sampler):
    """Threaded sampler that advances the plant and emits Samples, like pyhut's
    ThreadedSampler but driven by the sim plant."""

    def __init__(self, state, rate_hz=100.0):
        self._state = state
        self._period = 1.0 / rate_hz
        self._lock = threading.Lock()
        self._latest = None
        self._subs = []
        self._stop = threading.Event()
        self._thread = None
        self._last = None

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def latest(self):
        with self._lock:
            return self._latest

    def subscribe(self, callback):
        self._subs.append(callback)

        def _unsub():
            try:
                self._subs.remove(callback)
            except ValueError:
                pass
        return _unsub

    def _loop(self):
        self._last = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            dt = now - self._last
            self._last = now
            gyro_z, mag = self._state.advance(dt)
            s = Sample(
                t=now,
                accel=(0.0, 0.0, 1.0),
                gyro=(0.0, 0.0, gyro_z),
                temp_c=25.0,
                mag=mag,
                enc_left=self._state.enc_left,
                enc_right=self._state.enc_right,
                enc_left_rate=None,
                enc_right_rate=None,
                tof_mm=None,
                tof_t=None,
            )
            with self._lock:
                self._latest = s
            for cb in list(self._subs):
                try:
                    cb(s)
                except Exception:
                    pass
            self._stop.wait(self._period)

    def close(self):
        self.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()


class SimRobot(Robot):
    """A pyhut-shaped robot backed by the rotation plant.

    Construct with a :class:`SimConfig` to inject asymmetry/bias/sign/EMI. Expose
    the same accessors the primitive uses (``drivetrain``, ``sampler``,
    encoders) plus ``true_heading`` for test assertions.
    """

    def __init__(self, cfg=None, sampler_hz=100.0):
        self._cfg = cfg or SimConfig()
        self._state = _SimState(self._cfg)
        self._drivetrain = SimDrivetrain(self._state)
        self._left_enc = SimEncoder(self._state, "left")
        self._right_enc = SimEncoder(self._state, "right")
        self._sampler = SimSampler(self._state, rate_hz=sampler_hz)

    @property
    def drivetrain(self):
        return self._drivetrain

    @property
    def sampler(self):
        return self._sampler

    @property
    def left_encoder(self):
        return self._left_enc

    @property
    def right_encoder(self):
        return self._right_enc

    @property
    def imu(self):
        return None

    @property
    def magnetometer(self):
        return None

    @property
    def range_sensor(self):
        return None

    @property
    def display(self):
        return None

    @property
    def true_heading(self):
        return self._state.true_heading

    def close(self):
        self._sampler.close()
        self._drivetrain.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
