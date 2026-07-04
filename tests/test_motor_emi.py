"""Tests for the motor-EMI diagnostic. No hardware: uses the synthetic robot."""

import json
import math
import os

from duckiebot_lab.experiments import motor_emi as m
from duckiebot_lab.experiments._sim import SimRobot


# ---------------------------------------------------------------------------
# Pure math helpers
# ---------------------------------------------------------------------------
def test_worst_heading_error_bounds():
    # EMI perpendicular and equal to a quarter of H -> asin(0.25) ~ 14.5 deg.
    assert abs(m._worst_heading_error_deg(5.0, 20.0) - math.degrees(math.asin(0.25))) < 1e-6
    # EMI >= H saturates at 180 deg (the estimate can point anywhere).
    assert m._worst_heading_error_deg(25.0, 20.0) == 180.0
    # No horizontal field -> undefined.
    assert math.isnan(m._worst_heading_error_deg(1.0, 0.0))


def test_angular_diff_wraps():
    assert abs(m._ang_diff_deg(359.0, 1.0) - (-2.0)) < 1e-9
    assert abs(m._ang_diff_deg(1.0, 359.0) - 2.0) < 1e-9


def test_circular_mean_across_wrap():
    # Mean of 350 and 10 is 0, not 180.
    assert abs(m._ang_diff_deg(m._circular_mean_deg([350.0, 10.0]), 0.0)) < 1e-6


# ---------------------------------------------------------------------------
# analyze(): synthetic windows -> level results referenced to baselines
# ---------------------------------------------------------------------------
def _win(label, pattern, duty, mag, gyro_rms=0.2):
    """Build a WindowStats with the derived mag stats filled in."""
    mx, my, mz = mag
    fm = math.sqrt(mx * mx + my * my + mz * mz)
    return m.WindowStats(
        label=label, pattern=pattern, duty=duty,
        t0=0.0, t1=1.0, n=50, rate_hz=50.0,
        mag_mean=mag, field_mag_mean=fm, field_mag_std=0.1,
        heading_mean=m._heading_deg(mx, my), heading_std=0.2,
        gyro_mean=(0.0, 0.0, 0.0), gyro_rms=gyro_rms, mag_ok=True)


def test_analyze_recovers_emi_vector():
    base = (20.0, 0.0, 40.0)          # heading 0, |H| = 20
    # A pure +y EMI of 4 uT while motors run -> perpendicular to H.
    on = (20.0, 4.0, 40.0)
    windows = [
        _win("baseline", "off", 0.0, base),
        _win("d=0.50", "spin", 0.5, on),
        _win("baseline", "off", 0.0, base),
    ]
    levels, noise = m.analyze(windows)
    assert len(levels) == 1
    lv = levels[0]
    assert abs(lv.emi_horiz - 4.0) < 1e-6
    # Measured shift and worst-case coincide here (EMI is perpendicular to H).
    assert abs(lv.heading_err_meas - math.degrees(math.atan2(4.0, 20.0))) < 1e-6
    assert abs(lv.heading_err_worst - math.degrees(math.asin(4.0 / 20.0))) < 1e-6
    assert noise["n_off_windows"] == 2


def test_verdict_static_only_when_over_threshold_everywhere():
    base = (20.0, 0.0, 40.0)
    # Big EMI even at the lowest duty -> STATIC_ONLY.
    windows = [
        _win("baseline", "off", 0.0, base),
        _win("d=0.10", "spin", 0.10, (20.0, 6.0, 40.0)),
        _win("baseline", "off", 0.0, base),
        _win("d=0.40", "spin", 0.40, (20.0, 12.0, 40.0)),
        _win("baseline", "off", 0.0, base),
    ]
    levels, noise = m.analyze(windows)
    v = m.decide_verdict(levels, threshold_deg=5.0, baseline_noise=noise)
    assert v.status == "STATIC_ONLY"
    assert v.mag_live_usable is False


def test_verdict_live_ok_when_small():
    base = (20.0, 0.0, 40.0)
    windows = [
        _win("baseline", "off", 0.0, base),
        _win("d=0.10", "spin", 0.10, (20.0, 0.2, 40.0)),
        _win("baseline", "off", 0.0, base),
        _win("d=0.50", "spin", 0.50, (20.0, 0.6, 40.0)),
        _win("baseline", "off", 0.0, base),
    ]
    levels, noise = m.analyze(windows)
    v = m.decide_verdict(levels, threshold_deg=5.0, baseline_noise=noise)
    assert v.status == "LIVE_OK"
    assert v.mag_live_usable is True


def test_verdict_marginal_reports_safe_duty():
    base = (20.0, 0.0, 40.0)
    # Small at 0.1, large at 0.6 -> crosses somewhere between: MARGINAL.
    windows = [
        _win("baseline", "off", 0.0, base),
        _win("d=0.10", "spin", 0.10, (20.0, 0.5, 40.0)),
        _win("baseline", "off", 0.0, base),
        _win("d=0.60", "spin", 0.60, (20.0, 6.0, 40.0)),
        _win("baseline", "off", 0.0, base),
    ]
    levels, noise = m.analyze(windows)
    v = m.decide_verdict(levels, threshold_deg=5.0, baseline_noise=noise)
    assert v.status == "MARGINAL"
    assert v.mag_live_usable is True
    assert v.max_safe_duty is not None
    assert 0.10 < v.max_safe_duty < 0.60


def test_operating_duty_pins_the_call():
    base = (20.0, 0.0, 40.0)
    windows = [
        _win("baseline", "off", 0.0, base),
        _win("d=0.20", "spin", 0.20, (20.0, 1.0, 40.0)),   # ~2.9 deg
        _win("baseline", "off", 0.0, base),
        _win("d=0.60", "spin", 0.60, (20.0, 6.0, 40.0)),   # ~16.7 deg
        _win("baseline", "off", 0.0, base),
    ]
    levels, noise = m.analyze(windows)
    v_low = m.decide_verdict(levels, threshold_deg=5.0, operating_duty=0.2,
                             baseline_noise=noise)
    assert v_low.status == "LIVE_OK"
    v_high = m.decide_verdict(levels, threshold_deg=5.0, operating_duty=0.6,
                              baseline_noise=noise)
    assert v_high.status in ("MARGINAL", "STATIC_ONLY")


# ---------------------------------------------------------------------------
# End-to-end against the synthetic robot (real threads, short windows)
# ---------------------------------------------------------------------------
def test_end_to_end_bench_sim():
    robot = SimRobot(mode="bench", imu_rate_hz=100.0)
    try:
        report = m.characterize(
            robot, duties=(0.1, 0.3, 0.6, 0.9),
            settle_s=0.05, dwell_s=0.25, baseline_s=0.25,
            threshold_deg=5.0)
    finally:
        robot.close()

    # We should have one level per duty, monotone-ish rising worst-case error.
    assert len(report.levels) == 4
    worst = [lv.heading_err_worst for lv in report.levels]
    assert worst[0] < worst[-1]
    # Bench mode: gyro is ~0, so nothing should be flagged as rotating.
    assert not any(lv.rotating for lv in report.levels)
    # The tuned sim crosses 5 deg in-range -> MARGINAL with a sane safe duty.
    assert report.verdict.status == "MARGINAL"
    assert 0.1 < report.verdict.max_safe_duty < 0.9


def test_end_to_end_floor_sim_flags_rotation():
    robot = SimRobot(mode="floor", imu_rate_hz=100.0)
    try:
        report = m.characterize(
            robot, duties=(0.4, 0.8),
            settle_s=0.05, dwell_s=0.25, baseline_s=0.25)
    finally:
        robot.close()
    # Floor mode really spins the chassis, so the gyro cross-check trips.
    assert any(lv.rotating for lv in report.levels)
    # |B| deviation stays a valid EMI signal under rotation.
    assert all(not math.isnan(lv.field_mag_dev_pct) for lv in report.levels)


def test_save_report_writes_files(tmp_path):
    robot = SimRobot(mode="bench", imu_rate_hz=100.0)
    try:
        report = m.characterize(robot, duties=(0.2, 0.6),
                                settle_s=0.05, dwell_s=0.2, baseline_s=0.2)
    finally:
        robot.close()
    paths = m.save_report(report, outdir=str(tmp_path), stem="run1")
    assert os.path.exists(paths["json"])
    assert os.path.exists(paths["csv"])
    assert os.path.exists(paths["txt"])
    with open(paths["json"]) as f:
        data = json.load(f)
    # The machine-readable verdict Stage 4 keys off must be present.
    assert data["verdict"]["status"] in ("LIVE_OK", "MARGINAL", "STATIC_ONLY")
    assert isinstance(data["verdict"]["mag_live_usable"], bool)
    assert data["schema"] == m.EMIReport.SCHEMA


# ---------------------------------------------------------------------------
# Regression tests for the verdict-safety fixes.
#
# Both bugs below let the tool emit LIVE_OK on data that is actually over
# budget. They are grounded in real runs observed on-hardware, where repeating
# the same duty gave worst-case errors of 5.6 .. 10.1 deg against a 5 deg budget
# yet an early version reported LIVE_OK. The verdict must key off the WORST
# observed error per duty, and a large at-rest noise floor must never be a
# licence to fuse.
# ---------------------------------------------------------------------------

def _lvl(duty, worst, rotating=False):
    """Minimal LevelResult carrying just the fields the verdict logic reads."""
    return m.LevelResult(
        duty=duty, pattern="spin", emi_vec=(0.0, 0.0, 0.0),
        emi_mag=float("nan"), emi_horiz=float("nan"),
        heading_err_meas=float("nan"), heading_err_worst=worst,
        field_mag_dev_pct=float("nan"), gyro_rms=1.0, rotating=rotating,
        baseline_field_mag=15.0, baseline_horiz=12.0,
    )


def test_repeated_duty_uses_worst_not_lucky_sample():
    # Five repeats of duty 0.5: worst-case 9.7/8.3/7.1/10.1/5.6 deg, all >= budget.
    # A lucky low repeat must not produce LIVE_OK.
    levels = [_lvl(0.5, e) for e in (9.7, 8.3, 7.1, 10.1, 5.6)]
    bn = {"heading_std_deg": 3.70}
    v = m.decide_verdict(levels, threshold_deg=5.0, operating_duty=0.5,
                         baseline_noise=bn)
    assert v.status == "STATIC_ONLY"
    assert v.mag_live_usable is False
    # The reported operating-duty error is the WORST repeat, not the best.
    assert abs(v.error_at_operating_deg - 10.1) < 1e-6


def test_lucky_dip_at_operating_duty_is_overridden_by_repeat():
    # A single 0.6 deg reading at 0.5 sits beside a 7.0 deg repeat at 0.5;
    # the bad repeat must win so the verdict is not LIVE_OK.
    levels = [_lvl(0.1, 13.6), _lvl(0.1, 8.6), _lvl(0.3, 4.2),
              _lvl(0.5, 0.6), _lvl(0.5, 7.0), _lvl(0.8, 8.3)]
    v = m.decide_verdict(levels, threshold_deg=5.0, operating_duty=0.5,
                         baseline_noise={"heading_std_deg": 2.89})
    assert v.status == "STATIC_ONLY"
    assert v.error_at_operating_deg >= 7.0 - 1e-6


def test_high_noise_floor_never_upgrades_to_live_ok():
    # Over-budget effect (max 10.1) with a LARGE noise floor (3.7 deg) must NOT
    # be forgiven -- a noisy-at-rest mag is worse, not safer.
    levels = [_lvl(0.5, e) for e in (9.7, 8.3, 7.1, 10.1, 5.6)]
    v = m.decide_verdict(levels, threshold_deg=5.0, operating_duty=None,
                         baseline_noise={"heading_std_deg": 3.70})
    assert v.status == "STATIC_ONLY"
    assert v.mag_live_usable is False


def test_noise_gate_still_forgives_tiny_effect_with_tight_floor():
    # Genuinely small EMI (<= budget) measured against a TIGHT noise floor
    # should still pass -- the gate is one-directional, not a blanket fail.
    levels = [_lvl(0.1, 1.0), _lvl(0.3, 1.5), _lvl(0.5, 2.0), _lvl(0.8, 2.5)]
    v = m.decide_verdict(levels, threshold_deg=5.0, operating_duty=0.5,
                         baseline_noise={"heading_std_deg": 0.4})
    assert v.status == "LIVE_OK"
    assert v.mag_live_usable is True


def test_single_duty_trend_is_not_nan_in_text():
    # A single-duty sweep has no slope; the text report must not print 'nan'
    # and should surface the spread across repeats instead.
    levels = [_lvl(0.5, e) for e in (9.7, 8.3, 7.1, 10.1, 5.6)]
    report = m.EMIReport(
        windows=[],
        levels=levels,
        baseline_noise={"heading_std_deg": 3.70, "field_mag_std_uT": 1.6,
                        "field_mag_uT": 14.6, "horiz_uT": 16.4},
        verdict=m.decide_verdict(levels, threshold_deg=5.0, operating_duty=0.5,
                                 baseline_noise={"heading_std_deg": 3.70}),
        meta={},
    )
    text = m.format_report_text(report)
    # The trend line specifically must not degrade to 'nan' on a single duty;
    # it should report the spread across repeats instead. (Other table columns
    # are nan here only because this fixture populates just heading_err_worst.)
    trend_line = [ln for ln in text.splitlines() if ln.startswith("Trend:")][0]
    assert "nan" not in trend_line.lower()
    assert "spread" in trend_line.lower()
