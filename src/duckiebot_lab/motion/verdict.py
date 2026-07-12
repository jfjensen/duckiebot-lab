"""
Read the Stage 2b motor-EMI verdict and decide whether the magnetometer may be
fused as a *live* heading input during an in-place spin.

Stage 2b (``duckiebot_lab.experiments.motor_emi``) saves a JSON run whose
machine-readable ``verdict`` block is exactly this decision surface:

    {
      "verdict": {
        "status": "LIVE_OK" | "MARGINAL" | "STATIC_ONLY",
        "mag_live_usable": true | false,
        "max_safe_duty": 0.0 .. 1.0 | null,
        ...
      },
      ...
    }

Reading rules (from the Stage 2b write-up):
  * ``LIVE_OK``     -> the field is clean enough; fuse the mag live.
  * ``MARGINAL``    -> live-usable only if the spin stays at/under
                       ``max_safe_duty``; above that, don't.
  * ``STATIC_ONLY`` -> too much EMI; mag is a before/after reference only, never
                       fused during the turn.

Absent or unreadable verdict -> the safe default is gyro-only (don't fuse). The
primitive still works; it just relies on the pre-spin bias re-zero to keep the
gyro integral honest over one turn.
"""

import json
from collections import namedtuple


# fuse: bool -- may we fuse the mag live at the given operating duty?
# status: the raw verdict status (or None if no verdict).
# max_safe_duty: float or None -- the ceiling from a MARGINAL verdict.
# reason: short human string for logs/telemetry.
MagPolicy = namedtuple("MagPolicy", ["fuse", "status", "max_safe_duty", "reason"])


def load_emi_verdict(path):
    """Load the ``verdict`` block from a Stage 2b JSON run.

    Returns the verdict dict, or ``None`` if the file is missing/unreadable or
    carries no verdict block. Tolerant by design: a missing verdict must never
    crash a motion -- it just means "no live-mag clearance", which is safe.
    """
    if not path:
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (IOError, OSError, ValueError):
        return None
    verdict = data.get("verdict") if isinstance(data, dict) else None
    return verdict if isinstance(verdict, dict) else None


def decide_mag_fusion(verdict, operating_duty):
    """Map a verdict + the duty we intend to spin at to a :class:`MagPolicy`.

    ``operating_duty`` is the wheel-speed magnitude (0..1) the rotation will
    actually use, so a MARGINAL clearance is checked against the real operating
    point rather than assumed.
    """
    if not verdict:
        return MagPolicy(False, None, None, "no EMI verdict; gyro-only")

    status = verdict.get("status")
    max_safe = verdict.get("max_safe_duty")
    # Honour an explicit boolean if the tool set one; it is the tool's own
    # summary of the status line.
    explicit = verdict.get("mag_live_usable")

    if status == "LIVE_OK" or explicit is True and status != "STATIC_ONLY":
        return MagPolicy(True, status, max_safe, "LIVE_OK: fuse mag live")

    if status == "MARGINAL":
        if max_safe is None:
            return MagPolicy(False, status, None,
                             "MARGINAL but no max_safe_duty; gyro-only")
        if operating_duty <= max_safe + 1e-9:
            return MagPolicy(True, status, max_safe,
                             "MARGINAL: duty %.3f <= max_safe %.3f; fuse"
                             % (operating_duty, max_safe))
        return MagPolicy(False, status, max_safe,
                         "MARGINAL: duty %.3f > max_safe %.3f; gyro-only"
                         % (operating_duty, max_safe))

    if status == "STATIC_ONLY":
        return MagPolicy(False, status, max_safe,
                         "STATIC_ONLY: mag is reference-only; gyro-only")

    return MagPolicy(False, status, max_safe,
                     "unknown status %r; gyro-only" % (status,))
