"""Tests for the closed-loop rotate primitive and its helpers. No hardware:
everything runs against the threaded sim robot in duckiebot_lab.motion._sim."""

import json
import math

import pytest

from duckiebot_lab.motion import (
    MotionPrimitives, RotateConfig, rotate,
    wrap180, wrap360, ang_diff,
    load_emi_verdict, decide_mag_fusion,
)
from duckiebot_lab.motion._sim import SimRobot, SimConfig


# ---------------------------------------------------------------------------
# fast config so the suite doesn't spend seconds on bias windows
# ---------------------------------------------------------------------------
def fast_cfg(**over):
    c = RotateConfig()
    c.loop_hz = 100.0
    c.bias_estimate_s = 0.1
    c.probe_time_s = 0.15
    c.timeout_margin = 4.0
    for k, v in over.items():
        setattr(c, k, v)
    return c


# ---------------------------------------------------------------------------
# angle helpers
# ---------------------------------------------------------------------------
def test_wrap_helpers():
    assert wrap360(-90.0) == 270.0
    assert wrap360(450.0) == 90.0
    assert abs(wrap180(190.0) - (-170.0)) < 1e-9
    assert abs(wrap180(-190.0) - 170.0) < 1e-9
    assert abs(ang_diff(359.0, 1.0) - (-2.0)) < 1e-9
    assert abs(ang_diff(1.0, 359.0) - 2.0) < 1e-9


# ---------------------------------------------------------------------------
# verdict policy branches
# ---------------------------------------------------------------------------
def test_verdict_live_ok_fuses():
    p = decide_mag_fusion({"status": "LIVE_OK", "mag_live_usable": True,
                           "max_safe_duty": None}, operating_duty=0.5)
    assert p.fuse is True


def test_verdict_marginal_respects_max_safe_duty():
    v = {"status": "MARGINAL", "mag_live_usable": False, "max_safe_duty": 0.4}
    assert decide_mag_fusion(v, operating_duty=0.35).fuse is True
    assert decide_mag_fusion(v, operating_duty=0.55).fuse is False


def test_verdict_static_only_and_missing():
    assert decide_mag_fusion({"status": "STATIC_ONLY"}, 0.1).fuse is False
    assert decide_mag_fusion(None, 0.1).fuse is False


def test_load_verdict_missing_file(tmp_path):
    assert load_emi_verdict(str(tmp_path / "nope.json")) is None


def test_load_verdict_reads_block(tmp_path):
    p = tmp_path / "emi.json"
    p.write_text(json.dumps({"verdict": {"status": "LIVE_OK",
                                         "mag_live_usable": True,
                                         "max_safe_duty": None}}))
    v = load_emi_verdict(str(p))
    assert v["status"] == "LIVE_OK"


# ---------------------------------------------------------------------------
# direction auto-detect
# ---------------------------------------------------------------------------
def test_direction_autodetect_both_signs():
    for sign in (1, -1):
        cfg = SimConfig(); cfg.plant_sign = sign
        with SimRobot(cfg) as bot:
            mp = MotionPrimitives(bot)
            d = mp.calibrate_direction(fast_cfg())
            assert d in (1, -1)
            # commanding +heading with this sign must actually raise true heading
            bot.true_heading  # noqa
            r = mp.rotate(30.0, rate_dps=60.0, cfg=fast_cfg())
            assert bot.true_heading > 5.0   # turned the correct way


# ---------------------------------------------------------------------------
# core rotate behaviour
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("angle", [45.0, 90.0, -90.0])
def test_rotate_reaches_target(angle):
    with SimRobot() as bot:
        mp = MotionPrimitives(bot)
        r = mp.rotate(angle, rate_dps=60.0, cfg=fast_cfg())
        assert r.reached, r.reason
        # fused heading landed within tolerance of the request
        assert abs(r.error_deg) <= 2.5
        # and ground truth agrees with the fused estimate to a few degrees
        assert abs(bot.true_heading - r.achieved_deg) < 6.0
        # sign is right
        assert math.copysign(1, bot.true_heading) == math.copysign(1, angle)


def test_rotate_compensates_asymmetry():
    # Exaggerate the right-side weakness; the inner rate loop must still converge.
    cfg = SimConfig(); cfg.right_gain = 0.6
    with SimRobot(cfg) as bot:
        mp = MotionPrimitives(bot)
        r = mp.rotate(90.0, rate_dps=60.0, cfg=fast_cfg())
        assert r.reached, r.reason
        assert abs(r.error_deg) <= 3.0


def test_rotate_slow_gives_more_samples_per_degree():
    with SimRobot() as bot:
        mp = MotionPrimitives(bot, dir_sign=None)
        slow = mp.rotate(90.0, rate_dps=30.0, cfg=fast_cfg())
    with SimRobot() as bot2:
        mp2 = MotionPrimitives(bot2)
        fast = mp2.rotate(90.0, rate_dps=90.0, cfg=fast_cfg())
    slow_density = slow.n_samples / abs(slow.achieved_deg or 1)
    fast_density = fast.n_samples / abs(fast.achieved_deg or 1)
    assert slow_density > fast_density


def test_rotate_explicit_dir_sign_skips_probe():
    with SimRobot() as bot:
        # give the correct sign explicitly (plant_sign=+1 -> needs dir_sign=-1)
        mp = MotionPrimitives(bot, dir_sign=-1)
        r = mp.rotate(60.0, rate_dps=60.0, cfg=fast_cfg())
        assert r.reached
        assert r.dir_sign == -1


def test_rotate_encoder_crosscheck_tracks():
    with SimRobot() as bot:
        mp = MotionPrimitives(bot)
        r = mp.rotate(90.0, rate_dps=60.0, cfg=fast_cfg())
        # encoders are magnitude-only but geometry-consistent here, so the
        # magnitude should be in the ballpark of the turn.
        assert r.enc_magnitude_deg > 45.0
        assert r.enc_disagreement < 30.0


def test_rotate_mag_fused_when_policy_allows():
    v = {"status": "LIVE_OK", "mag_live_usable": True, "max_safe_duty": None}
    policy = decide_mag_fusion(v, 0.5)
    cfg = SimConfig(); cfg.emi_uT = 0.0     # clean field -> mag agrees with gyro
    with SimRobot(cfg) as bot:
        mp = MotionPrimitives(bot, mag_policy=policy)
        r = mp.rotate(90.0, rate_dps=60.0, cfg=fast_cfg())
        assert r.mag_fused is True
        assert r.reached


def test_rotate_gyro_only_when_no_policy():
    with SimRobot() as bot:
        mp = MotionPrimitives(bot)
        r = mp.rotate(45.0, rate_dps=60.0, cfg=fast_cfg())
        assert r.mag_fused is False


def test_rotate_times_out_when_stalled_without_hanging():
    # Deadband above the stiction floor -> wheels never move -> must time out,
    # not spin forever.
    cfg = SimConfig(); cfg.deadband = 0.95
    with SimRobot(cfg) as bot:
        mp = MotionPrimitives(bot, dir_sign=-1)   # skip the (also-stalled) probe
        c = fast_cfg(); c.min_move_base = 0.3     # below deadband on purpose
        c.timeout_margin = 1.0                    # keep the test quick
        r = mp.rotate(90.0, rate_dps=60.0, cfg=c)
        assert r.reached is False
        assert "timeout" in r.reason
        assert r.duration_s < 6.0


def test_bias_rezero_removes_offset():
    # A large constant gyro bias must be measured out; otherwise the integral
    # would run away and the turn would badly overshoot.
    cfg = SimConfig(); cfg.gyro_bias_dps = 8.0
    with SimRobot(cfg) as bot:
        mp = MotionPrimitives(bot)
        r = mp.rotate(90.0, rate_dps=60.0, cfg=fast_cfg())
        assert abs(r.gyro_bias_dps - 8.0) < 2.0   # recovered the bias
        assert r.reached
        assert abs(r.error_deg) <= 3.0


def test_convenience_rotate_wrapper(tmp_path):
    p = tmp_path / "emi.json"
    p.write_text(json.dumps({"verdict": {"status": "LIVE_OK",
                                         "mag_live_usable": True,
                                         "max_safe_duty": None}}))
    with SimRobot() as bot:
        r = rotate(bot, 60.0, rate_dps=60.0, verdict_path=str(p), cfg=fast_cfg())
        assert r.reached
        assert r.mag_fused is True
