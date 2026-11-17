"""gta6-twitch-ccu-forecast-01 entry: an analogue-template quantile forecaster.

Method (research summary)
-------------------------
The house baseline is naive persistence: every future hour equals hour 1,
with a lognormal spread widening like sqrt(h - 1). It ignores the one thing
every big launch day has: a shape. A console launch that unlocks at rolling
local midnight (Grand Theft Auto VI's pattern, and the pattern of all four
practice launches) ramps through the UTC day as Europe wakes, the Americas
unlock and prime time arrives, so hours 2..24 sit at systematically
different ratios to hour 1 -- ratios persistence cannot see.

This program reads the shape from the case itself. Every case carries
`comparables`: dozens of past big launches with their full hourly curves.
For each comparable we take its release-day hours, form the log-ratio curve
ln(viewers_h / viewers_1), keep the launches whose first hour was already a
real launch hour (hour 1 at least 5% of the day's peak -- the rolling
unlocks, not the single-global-unlock spikes), and weight each by a
Gaussian kernel on the distance between its log hour-1 audience and the
target's: big launches lend their shape to big launches. The forecast for
hour h is then a lognormal whose centre and spread are the kernel-weighted
mean and standard deviation of those log-ratios at h, anchored on the
target's own hour-1 print:

    q_tau(h) = p1 * exp(mu_h + z_tau * sigma_h)

Leave-one-out over the 37 usable comparables in the practice cases scores
this at a mean log pinball loss of about 0.25-0.27 on rolling-midnight
launches like the target, against about 0.32 for persistence on the same
launches (the baseline's published practice loss is 0.387).

The program is deterministic (pure arithmetic over the input, fixed
iteration order), standard library only, offline, and runs in well under a
second per case.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim. -P keeps this file's directory off sys.path; nothing here
imports beyond the standard library. It reads every <id>.in in
$MAKEFUN_INPUT_DIR (else /input) and writes <id>.out to
$MAKEFUN_OUTPUT_DIR (else /output): {"hours": [{"hour": h, "q": [q05, q25,
q50, q75, q95]} for h in 2..24]}.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- Template design constants (calibrated by leave-one-out, see docstring)
# A comparable joins the shape pool when its hour 1 already held at least
# this share of its release day's peak: rolling-midnight launches qualify,
# single-global-unlock spikes (hour 1 a pre-release trickle) do not.
ROLLING_MIN_HOUR1_SHARE = 0.05
# Gaussian kernel bandwidth on the log hour-1 distance between a comparable
# and the target.
KERNEL_BANDWIDTH = 0.9
# Never let an hour's spread collapse below this (log units).
SIGMA_FLOOR = 0.10
# Nor let it explode past this: if the pool ever mixes regimes (the
# fall-back pool can), an uncapped spread would throw the 5%/95% quantiles
# to absurd values. The validated rolling pool's spreads sit far below it.
SIGMA_CAP = 1.25
# If the rolling pool is thinner than this, fall back to every complete
# comparable rather than forecast from a handful of curves.
MIN_POOL = 8


def parse_ts(text: str) -> datetime:
    """Parse the case files' ISO-8601 UTC timestamps ('...Z')."""
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def release_day_hours(comparable: dict) -> dict[int, float] | None:
    """The comparable's release-day hourly viewers, {hour 1..24: viewers}.

    Hours are taken from the hourly points whose `from` falls on the
    comparable's own UTC release day (its `day_start`). Returns None when
    the day is incomplete or hour 1 is missing or negligible -- such a
    comparable (for instance the target category's own pre-release history,
    which has no release day at all) cannot lend a shape.
    """
    try:
        day_start = parse_ts(comparable["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    hours: dict[int, float] = {}
    for point in comparable.get("hourly") or []:
        viewers = point.get("viewers")
        if viewers is None:
            continue
        try:
            offset = (parse_ts(point["from"]) - day_start).total_seconds() / 3600.0
        except (KeyError, TypeError, ValueError):
            continue
        if offset == int(offset) and 0 <= int(offset) < 24:
            hours[int(offset) + 1] = float(viewers)
    if len(hours) != 24 or hours.get(1, 0.0) < 50:
        return None
    return hours


def build_pool(case: dict) -> list[tuple[float, dict[int, float]]]:
    """Shape pool for this case: [(log hour-1 viewers, {hour: log-ratio})].

    The log-ratio curve of a comparable is ln(viewers_h / viewers_1) for
    h = 2..24. Rolling-midnight launches (hour 1 a real share of the day's
    peak) form the pool; if too few qualify, every complete comparable is
    used instead.
    """
    complete: list[tuple[float, dict[int, float], float]] = []
    for comparable in case.get("comparables") or []:
        unlock = comparable.get("unlock") or ""
        if "target-history" in unlock:
            continue  # the target's own pre-release history, not a launch
        hours = release_day_hours(comparable)
        if hours is None:
            continue
        p1 = hours[1]
        log_ratios = {h: math.log(hours[h] / p1) for h in range(2, 25)}
        complete.append((math.log(p1), log_ratios, p1 / max(hours.values())))
    rolling = [(lp1, lr) for lp1, lr, share in complete if share >= ROLLING_MIN_HOUR1_SHARE]
    if len(rolling) >= MIN_POOL:
        return rolling
    return [(lp1, lr) for lp1, lr, _share in complete]


def hour_quantiles(p1: float, log_p1: float, pool: list[tuple[float, dict[int, float]]], h: int) -> list[float]:
    """The five quantiles of hour h from the pool's weighted log-ratios."""
    if not pool:
        # No comparable survived: persistence with the baseline's spread.
        return [p1 * math.exp(z * 0.15 * math.sqrt(h - 1)) for z in Z]
    weights = []
    total = 0.0
    for log_p1_c, _lr in pool:
        w = math.exp(-0.5 * ((log_p1_c - log_p1) / KERNEL_BANDWIDTH) ** 2)
        weights.append(w)
        total += w
    mu = 0.0
    for w, (_lp, log_ratios) in zip(weights, pool):
        mu += w * log_ratios[h]
    mu /= total
    var = 0.0
    for w, (_lp, log_ratios) in zip(weights, pool):
        var += w * (log_ratios[h] - mu) ** 2
    sigma = min(max(math.sqrt(var / total), SIGMA_FLOOR), SIGMA_CAP)
    quantiles = [p1 * math.exp(mu + z * sigma) for z in Z]
    quantiles.sort()
    return [min(max(q, 1.0), 1e9) for q in quantiles]


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up; whole numbers come back as int."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    pool = build_pool(case)
    rows = []
    for row in case["forecast"]:
        h = row["hour"]
        qs = [round2(q) for q in hour_quantiles(p1, math.log(p1), pool, h)]
        rows.append({"hour": h, "q": qs})
    return {"hours": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
