# Motion primitives (`duckiebot_lab.motion`)

Closed-loop, app-side control policy built on pyhut's ABCs. The first primitive
is `rotate(angle_deg, rate_dps)`: an in-place turn that spins at a commanded
angular rate and stops at a target angle using **measured** heading, not an
open-loop "spin for N seconds". Structured so `drive_straight` can follow.

## Why it lives here, not in pyhut

pyhut's drivetrain is deliberately dumb and safe: you set wheel speeds and a
0.3 s watchdog zeros the motors if you stop feeding it. It has no notion of "turn
90 degrees". Closing a loop around measured heading to hit an angle is *policy*,
so it belongs in the application, written against pyhut's interfaces so any robot
implementing them is drivable by the same code. The dependency arrow stays
one-way: `duckiebot-lab -> pyhut`.

## The heading estimate

Feedback comes from the Stage 1 Sampler, fused in the order the hardware forces:

1. **Primary: integrated, bias-corrected gyro-z.** `Sample.gyro[2]` is deg/s;
   when the robot is built with a pyhut `Calibration`, the sampler is wired over
   the calibrated IMU view, so that value already has the gyro bias removed. The
   primitive additionally re-zeros the *residual* bias in the ~0.4 s before each
   turn, because a gyro's zero-rate output drifts with temperature and the
   freshest offset is the honest one. Heading is the corrected rate integrated
   over the measured `dt` between consecutive samples.

2. **Conditional: magnetometer, complementary-filtered.** The mag is an absolute
   reference that stops the gyro integral drifting over a long turn, but only if
   Stage 2b's EMI verdict clears it *at the duty this turn runs at*. It is fused
   as a slow correction (weight `1 - alpha`, `alpha = 0.98`), and its rotation
   sense is **reconciled against the gyro** before it is trusted (see below), so
   an inverted compass axis can't corrupt the estimate.

3. **Cross-check only: encoders as magnitude.** The Hall encoders count edges
   up-only and cannot sense direction, so they never steer. Their tick deltas
   become an independent *magnitude* of rotation (with the commanded sign
   applied) purely to catch a stalled wheel, gross slip, or a dead gyro. Surfaced
   as `enc_magnitude_deg` and `enc_disagreement` on the result.

## Two signs are measured, never assumed

Two hardware facts depend on mounting/wiring and are auto-detected rather than
hard-coded:

* **Wheel/gyro direction.** Which wheel pairing (`+left,-right` vs `-left,+right`)
  makes the measured heading *increase* is found by a brief probe
  (`calibrate_direction`) that nudges one way, reads the sign of the measured
  rate, then nudges back an equal amount so the robot ends where it started. The
  sign is cached; pass `dir_sign` to skip the probe. For a multi-turn sequence
  (the lidar demo), probe once up front.

* **Magnetometer rotation sense.** When mag fusion is on, the tracker waits until
  ~10 deg of gyro rotation has accumulated, then locks the mag's sign to whatever
  agrees with the gyro. Until then it runs gyro-only. Same "measure, don't
  assume" idea as the wheel sign.

## Compensating the asymmetric drive paths

Left is all-PCA channels; right is GPIO direction + PCA PWM, and the driver
applies no per-side trim. The controller is a cascade:

* **Outer / position loop:** heading error -> a target angular rate, saturated at
  `rate_dps` and tapered as the error shrinks, so the robot decelerates into the
  target instead of overshooting.
* **Inner / rate loop:** a PI-plus-feed-forward loop drives the *measured* rate
  to that target by setting the base wheel magnitude. Because it closes on
  measured rate, it absorbs the left/right mismatch: if one side is weaker for a
  given command, the rate falls short and the loop pushes harder. `left_trim` /
  `right_trim` give it a feed-forward head start if you know the imbalance.

Near the target the controller never brakes by reversing while still short (that
plus the stiction floor would set up a limit cycle); it coasts in, then cuts
power inside the tolerance band and lets the rate fall so the "stopped" test can
trigger. Every tick calls `set_target`, so the watchdog never trips mid-turn.

Slow is the default (`rate_dps=45`): less scrub/slip, more samples per degree,
less measurement smear.

## Usage

```python
from pyhut import DuckiebotHUT
from duckiebot_lab.motion import MotionPrimitives, decide_mag_fusion, load_emi_verdict

# Build with the SAME calibration Stages 4/5 run with, so the sampler's gyro/mag
# are the calibrated values the loop expects.
with DuckiebotHUT(calibration="calib.json") as bot:
    policy = decide_mag_fusion(load_emi_verdict("emi_run.json"), operating_duty=0.35)
    mp = MotionPrimitives(bot, mag_policy=policy)
    mp.calibrate_direction()             # once, up front (optional; else lazy)

    result = mp.rotate(90.0, rate_dps=45.0)
    print(result.achieved_deg, result.error_deg, result.reached)
```

One-shot convenience:

```python
from duckiebot_lab.motion import rotate
r = rotate(bot, 90.0, rate_dps=45.0, verdict_path="emi_run.json")
```

CLI / smoke test (no hardware needed):

```bash
duckiebot-rotate --simulate --angle 90 --rate 45
duckiebot-rotate --simulate --angle -120 --plant-sign -1 --emi live
# on the robot:
sudo duckiebot-rotate --angle 90 --rate 45 --calibration calib.json \
                      --emi-verdict emi_run.json
```

## `RotateResult` fields

`requested_deg`, `achieved_deg`, `error_deg`, `reached`, `duration_s`,
`n_samples`, `mag_fused`, `dir_sign`, `enc_magnitude_deg`, `enc_disagreement`,
`gyro_bias_dps`, `reason`.

## Measure these on your bot

`track_width_m` (default 0.10) and `wheel_diameter_m` (default 0.067) only feed
the encoder magnitude cross-check; they do **not** affect the gyro-primary
steering. Measure them so the cross-check means something. `ticks_per_rev` is
read from the encoder (pyhut's Duckiebot: 135).

A couple of honest limits: the turn stops at the near edge of the tolerance band
(default 1.5 deg), so it tends to undershoot by up to that much. That is fine for
the lidar demo, whose principle is "make the angle *known*, not the rotation
precise" -- every ToF sample is tagged with the measured heading regardless. And
`min_move_base` is a stiction floor: if you request a rate so slow that the floor
exceeds it, the bot turns at the floor rate instead. The rate is a target, not a
hard cap; the *angle* stays closed-loop-correct either way.

## Tests

```bash
pip install -e ".[test]"
pytest                       # or: python3 -m pytest   (on the Jetson)
```

19 tests run against a threaded synthetic robot (`motion/_sim.py`) that
implements pyhut's ABCs and injects drive asymmetry, a deadband, gyro bias +
noise, an arbitrary mounting sign, and an inverted-convention magnetometer -- so
the direction auto-detect, asymmetry compensation, mag-sign reconciliation, and
bias re-zero are all actually exercised, not assumed.
