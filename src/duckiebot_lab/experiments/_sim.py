"""
Synthetic robot for exercising the motor-EMI diagnostic without hardware.

It implements pyhut's ABCs (Drivetrain, Sampler, Robot-ish surface) and injects
a physically-shaped disturbance:

    measured_field = R(theta) . earth   +   emi(duty)   +   noise

* ``earth`` has constant magnitude; only its sensor-frame direction changes with
  chassis yaw ``theta``.
* ``emi(duty)`` is a fixed-direction vector in the *sensor* frame whose size
  grows with mean motor duty -- exactly the "motor-fixed offset that appears only
  when the motors run" the real EMI behaves like.

Two modes:
* ``bench`` -- chassis held still (wheels free on a stand): theta stays 0, gyro
  ~ 0, so the EMI shows up directly as a measured heading shift.
* ``floor`` -- an in-place spin really turns the chassis: theta integrates the
  commanded spin rate, gyro reports it, and the diagnostic must fall back to the
  |B|-deviation / worst-case columns.

This is a *test double*, not a hardware path -- it never touches a bus.
"""

import math
import random
import threading
import time

from pyhut.interfaces import Drivetrain, Sampler, Sample


# Synthetic environment constants (roughly realistic magnitudes in uT / dps).
_EARTH_H = 22.0          # horizontal Earth field magnitude
_EARTH_Z = 40.0          # vertical component
_BASE_BEARING = 35.0     # sensor-frame heading with motors off, bench mode
_EMI_DIR_DEG = 115.0     # fixed sensor-frame direction of the EMI horizontal part
_EMI_H_GAIN = 6.5        # horizontal EMI uT per unit mean duty
_EMI_Z_GAIN = 2.0        # vertical EMI uT per unit mean duty
_SPIN_GAIN_DPS = 350.0   # floor-mode yaw rate per unit spin command
_MAG_NOISE = 0.15        # per-axis mag noise (uT), ~ the AK8963 quantum
_GYRO_NOISE = 0.3        # per-axis gyro noise (dps) at rest


class SimDrivetrain(Drivetrain):
    """Records the commanded target into shared state; no watchdog needed."""

    def __init__(self, state):
        self._state = state

    def set_target(self, left, right):
        left = max(-1.0, min(1.0, left))
        right = max(-1.0, min(1.0, right))
        with self._state["lock"]:
            self._state["left"] = left
            self._state["right"] = right

    def stop(self):
        self.set_target(0.0, 0.0)

    def close(self):
        self.stop()


class SimSampler(Sampler):
    """Generates Samples from the synthetic model at a fixed rate."""

    def __init__(self, state, mode="bench", imu_rate_hz=50.0):
        self._state = state
        self._mode = mode
        self._period = 1.0 / imu_rate_hz if imu_rate_hz > 0 else 0.02
        self._lock = threading.Lock()
        self._sub_lock = threading.Lock()
        self._subs = []
        self._latest = None
        self._theta = 0.0                # chassis yaw, deg (floor mode)
        self._stop = threading.Event()
        self._thread = None
        self._rng = random.Random(1234)

    # -- Sampler ABC --------------------------------------------------------
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
        with self._sub_lock:
            self._subs.append(callback)

        def _unsub():
            with self._sub_lock:
                try:
                    self._subs.remove(callback)
                except ValueError:
                    pass
        return _unsub

    # -- model --------------------------------------------------------------
    def _loop(self):
        prev = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            dt = now - prev
            prev = now
            with self._state["lock"]:
                left = self._state["left"]
                right = self._state["right"]

            mean_duty = (abs(left) + abs(right)) / 2.0

            # Chassis rotation: only in floor mode, driven by the spin command.
            if self._mode == "floor":
                omega = _SPIN_GAIN_DPS * (left - right) / 2.0
                self._theta += omega * dt
                gz = omega + self._rng.gauss(0.0, _GYRO_NOISE)
            else:
                omega = 0.0
                gz = self._rng.gauss(0.0, _GYRO_NOISE)
            gx = self._rng.gauss(0.0, _GYRO_NOISE)
            gy = self._rng.gauss(0.0, _GYRO_NOISE)

            # Earth field in the sensor frame at the current yaw.
            ang = math.radians(_BASE_BEARING - self._theta)
            ex = _EARTH_H * math.cos(ang)
            ey = _EARTH_H * math.sin(ang)
            ez = _EARTH_Z

            # EMI: fixed-direction sensor-frame vector, grows with duty.
            eh = _EMI_H_GAIN * mean_duty
            emx = eh * math.cos(math.radians(_EMI_DIR_DEG))
            emy = eh * math.sin(math.radians(_EMI_DIR_DEG))
            emz = _EMI_Z_GAIN * mean_duty

            mx = ex + emx + self._rng.gauss(0.0, _MAG_NOISE)
            my = ey + emy + self._rng.gauss(0.0, _MAG_NOISE)
            mz = ez + emz + self._rng.gauss(0.0, _MAG_NOISE)

            sample = Sample(
                t=now,
                accel=(0.0, 0.0, 1.0), gyro=(gx, gy, gz), temp_c=30.0,
                mag=(mx, my, mz),
                enc_left=None, enc_right=None,
                enc_left_rate=None, enc_right_rate=None,
                tof_mm=None, tof_t=None,
            )
            with self._lock:
                self._latest = sample
            with self._sub_lock:
                subs = list(self._subs)
            for cb in subs:
                try:
                    cb(sample)
                except Exception:
                    pass
            self._stop.wait(self._period)


class SimRobot(object):
    """Minimal Robot surface the experiment uses: drivetrain + sampler."""

    def __init__(self, mode="bench", imu_rate_hz=50.0):
        self._state = {"lock": threading.Lock(), "left": 0.0, "right": 0.0}
        self._drivetrain = SimDrivetrain(self._state)
        self._sampler = SimSampler(self._state, mode=mode, imu_rate_hz=imu_rate_hz)

    @property
    def drivetrain(self):
        return self._drivetrain

    @property
    def sampler(self):
        return self._sampler

    def close(self):
        try:
            self._sampler.stop()
        finally:
            self._drivetrain.close()
