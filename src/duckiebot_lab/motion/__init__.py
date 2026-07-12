"""
duckiebot_lab.motion -- closed-loop motion primitives built on pyhut's ABCs.

Public API:
    MotionPrimitives   -- rotate(angle_deg, rate_dps); drive_straight is a seam.
    RotateConfig       -- tunables for a rotate call.
    RotateResult       -- structured outcome (achieved angle, error, diagnostics).
    HeadingTracker     -- gyro-primary heading fusion (mag/encoder as configured).
    load_emi_verdict,
    decide_mag_fusion,
    MagPolicy          -- read Stage 2b's EMI verdict into a mag-fusion policy.
    rotate             -- convenience one-shot wrapper.
"""

from .primitives import MotionPrimitives, RotateConfig, RotateResult
from .heading import HeadingTracker, estimate_gyro_bias, wrap360, wrap180, ang_diff
from .verdict import load_emi_verdict, decide_mag_fusion, MagPolicy

__all__ = [
    "MotionPrimitives", "RotateConfig", "RotateResult",
    "HeadingTracker", "estimate_gyro_bias", "wrap360", "wrap180", "ang_diff",
    "load_emi_verdict", "decide_mag_fusion", "MagPolicy",
    "rotate",
]


def rotate(robot, angle_deg, rate_dps=45.0, verdict_path=None,
           dir_sign=None, cfg=None, **kwargs):
    """One-shot convenience: build a MotionPrimitives, wire the EMI verdict if a
    path is given, and rotate once. Returns a RotateResult.

    For repeated turns (e.g. the lidar demo), construct MotionPrimitives once so
    the direction sign and sampler stay warm.
    """
    policy = None
    if verdict_path is not None:
        op_duty = kwargs.pop("operating_duty", 0.35)
        policy = decide_mag_fusion(load_emi_verdict(verdict_path), op_duty)
    mp = MotionPrimitives(robot, mag_policy=policy, dir_sign=dir_sign, **kwargs)
    return mp.rotate(angle_deg, rate_dps=rate_dps, cfg=cfg)
