"""
Command-line demo / smoke test for the closed-loop rotate primitive.

On the robot (root, since pyhut talks to /dev/i2c-1 and sysfs GPIO):

    sudo duckiebot-rotate --angle 90 --rate 45 --calibration calib.json \
                          --emi-verdict emi_run.json

Without hardware (threaded sim plant), to see the behaviour and output shape:

    duckiebot-rotate --simulate --angle 90 --rate 45
    duckiebot-rotate --simulate --angle -180 --plant-sign -1 --emi live

The demo builds the robot with the same calibration Stages 4/5 use, reads the
Stage 2b EMI verdict to decide mag fusion, and prints the RotateResult.
"""

import argparse
import sys


def _build_real_robot(args):
    from pyhut import DuckiebotHUT
    return DuckiebotHUT(calibration=args.calibration)


def _build_sim_robot(args):
    from ._sim import SimRobot, SimConfig
    cfg = SimConfig()
    cfg.plant_sign = args.plant_sign
    if args.emi == "live":
        cfg.emi_uT = 2.0
    elif args.emi == "bad":
        cfg.emi_uT = 12.0
    return SimRobot(cfg)


def _mag_policy(args):
    from .verdict import load_emi_verdict, decide_mag_fusion
    if args.simulate and args.emi in ("live", "bad") and not args.emi_verdict:
        # Synthesise a verdict for the demo when none is supplied.
        status = "LIVE_OK" if args.emi == "live" else "STATIC_ONLY"
        v = {"status": status, "mag_live_usable": status == "LIVE_OK",
             "max_safe_duty": None}
        return decide_mag_fusion(v, args.operating_duty)
    return decide_mag_fusion(load_emi_verdict(args.emi_verdict), args.operating_duty)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Closed-loop in-place rotate demo.")
    ap.add_argument("--angle", type=float, default=90.0, help="degrees (signed)")
    ap.add_argument("--rate", type=float, default=45.0, help="deg/s (slow is good)")
    ap.add_argument("--calibration", default=None,
                    help="pyhut calibration JSON (same one Stages 4/5 use)")
    ap.add_argument("--emi-verdict", default=None,
                    help="Stage 2b EMI verdict JSON; enables live mag fusion")
    ap.add_argument("--operating-duty", type=float, default=0.35,
                    help="duty used to check a MARGINAL verdict")
    ap.add_argument("--dir-sign", type=int, default=None, choices=(1, -1),
                    help="skip the direction probe with a known sign")
    ap.add_argument("--simulate", action="store_true", help="use the sim plant")
    ap.add_argument("--plant-sign", type=int, default=1, choices=(1, -1),
                    help="(sim) mounting sign to exercise auto-detect")
    ap.add_argument("--emi", choices=("off", "live", "bad"), default="off",
                    help="(sim) inject magnetometer EMI")
    ap.add_argument("--stall-nudges", type=int, default=None,
                    help="live-stall recovery attempts before giving up "
                         "(wheels ticking but heading not moving, e.g. a "
                         "caught caster); default from RotateConfig")
    args = ap.parse_args(argv)

    from .primitives import MotionPrimitives, RotateConfig

    robot = _build_sim_robot(args) if args.simulate else _build_real_robot(args)
    policy = _mag_policy(args)
    cfg = RotateConfig()
    if args.stall_nudges is not None:
        cfg.stall_max_nudges = args.stall_nudges
    try:
        mp = MotionPrimitives(robot, mag_policy=policy, dir_sign=args.dir_sign)
        result = mp.rotate(args.angle, rate_dps=args.rate, cfg=cfg)
    finally:
        robot.close()

    print("rotate(%.1f deg @ %.0f dps)" % (args.angle, args.rate))
    print("  reached          : %s" % result.reached)
    print("  achieved         : %.2f deg" % result.achieved_deg)
    print("  error            : %.2f deg" % result.error_deg)
    print("  duration         : %.2f s" % result.duration_s)
    print("  samples          : %d" % result.n_samples)
    print("  dir_sign         : %+d" % result.dir_sign)
    print("  gyro bias removed: %.2f deg/s" % result.gyro_bias_dps)
    print("  mag fused live   : %s" % result.mag_fused)
    print("  enc magnitude    : %.1f deg" % result.enc_magnitude_deg)
    print("  enc disagreement : %.1f deg" % result.enc_disagreement)
    print("  stall nudges     : %d" % result.stall_nudges)
    print("  reason           : %s" % result.reason)
    if args.simulate:
        print("  (sim true heading: %.2f deg)" % robot.true_heading)
    return 0 if result.reached else 1


if __name__ == "__main__":
    sys.exit(main())
