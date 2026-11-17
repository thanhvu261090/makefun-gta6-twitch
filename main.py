#!/usr/bin/env python3
"""Forecast Twitch viewers for GTA VI launch day, hours 2..24.

Method: nearest-neighbour log-ratio forecaster.

A launch's shape is mostly decided by how much of its audience is already
awake and unlocked when the UTC day starts.  A huge hour-1 print means most
of the world is already playing, so the curve decays from there; a small
hour-1 print means only New Zealand and Asia are in yet, so the curve
surges later as Europe and then the Americas unlock.

So, for every comparable past launch, I express its release-day hourly
counts as log ratios to that launch's own hour-1 print.  Then I pick the
K comparables whose hour-1 print is closest in log space to this case's
hour-1 print, and for each hour take the empirical quantiles of those
neighbours' log ratios, widened slightly.  The median is the central
forecast; the spread is the honest disagreement between similar launches.

Usage: python3 main.py
  Reads $MAKEFUN_INPUT_DIR (default /input) for *.in case files and writes
  one <id>.out per case to $MAKEFUN_OUTPUT_DIR (default /output).
  Standard library only, no network, deterministic.
"""

import json
import math
import os
import sys
from datetime import datetime

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
K_NEIGHBOURS = 9
SPREAD_MULT = 1.1
MAX_VIEWERS = 1_000_000_000.0
HOURS = range(2, 25)


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def release_day_curve(comparable):
    """Hourly viewers for a comparable's release day, hours 1..24.

    Returns a list of 24 values (None where the source has no point).
    """
    start = parse_time(comparable["day_start"])
    by_hour = {}
    for point in comparable.get("hourly") or []:
        viewers = point.get("viewers")
        if viewers is None:
            continue
        try:
            when = parse_time(point["from"])
        except (KeyError, TypeError, ValueError):
            continue
        offset = (when - start).total_seconds() / 3600.0
        hour = int(offset) + 1
        if 1 <= hour <= 24:
            by_hour[hour] = float(viewers)
    return [by_hour.get(h) for h in range(1, 25)]


def empirical_quantile(sorted_values, tau):
    """Linear-interpolation quantile of a sorted list."""
    n = len(sorted_values)
    if n == 0:
        return None
    if n == 1:
        return sorted_values[0]
    pos = tau * (n - 1)
    low = int(math.floor(pos))
    high = min(low + 1, n - 1)
    frac = pos - low
    return sorted_values[low] * (1.0 - frac) + sorted_values[high] * frac


def neighbours_by_first_hour(case_first, curves, k):
    """Curves sorted by closeness of their hour-1 print to the case's.

    Each curve is (name, 24 hourly viewers).  Only curves with a positive
    hour-1 print and at least one later hour count.  Ties break on name so
    the order is deterministic.
    """
    scored = []
    for name, curve in curves:
        first = curve[0]
        if not first or first <= 0:
            continue
        if not any(v and v > 0 for v in curve[1:]):
            continue
        distance = abs(math.log(first) - math.log(case_first))
        scored.append((distance, name, curve))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [curve for _, _, curve in scored[:k]]


def baseline_quantiles(first, hour):
    """Wide fallback: stay near the first hour, spread growing with time."""
    sigma = 0.5 * math.sqrt(hour - 1)
    return [first * math.exp(zi * sigma) for zi in
            (-1.6448536269514722, -0.6744897501960817, 0.0,
             0.6744897501960817, 1.6448536269514722)]


def forecast_case(case):
    known = case.get("known") or []
    first = 1.0
    if known:
        viewers = known[0].get("viewers")
        if viewers and viewers > 0:
            first = float(viewers)

    curves = []
    for comp in case.get("comparables") or []:
        curve = release_day_curve(comp)
        curves.append((comp.get("name", ""), curve))

    neighbours = neighbours_by_first_hour(first, curves, K_NEIGHBOURS)

    rows = []
    for hour in HOURS:
        ratios = []
        for curve in neighbours:
            base = curve[0]
            value = curve[hour - 1]
            if base and base > 0 and value and value > 0:
                ratios.append(math.log(value / base))
        if len(ratios) >= 3:
            ratios.sort()
            median = empirical_quantile(ratios, 0.5)
            quantiles = []
            for tau in LEVELS:
                q = empirical_quantile(ratios, tau)
                q = median + (q - median) * SPREAD_MULT
                quantiles.append(first * math.exp(q))
        elif neighbours:
            # Too few neighbours report this hour: fall back to the
            # median of the ratios we do have, with the wide fallback
            # spread around it instead of around the first hour.
            if ratios:
                ratios.sort()
                median = empirical_quantile(ratios, 0.5)
                centre = first * math.exp(median)
            else:
                centre = first
            quantiles = [centre * math.exp(zi * 0.5 * math.sqrt(hour - 1))
                         for zi in (-1.6448536269514722, -0.6744897501960817,
                                    0.0, 0.6744897501960817, 1.6448536269514722)]
        else:
            quantiles = baseline_quantiles(first, hour)
        # Keep every quantile inside the allowed range and ordered.
        cleaned = []
        prev = 0.0
        for q in quantiles:
            q = max(1.0, min(MAX_VIEWERS, q))
            q = max(q, prev)
            cleaned.append(round(q, 2))
            prev = q
        rows.append({"hour": hour, "q": cleaned})
    return {"hours": rows}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(output_dir, exist_ok=True)
    names = sorted(
        name for name in os.listdir(input_dir) if name.endswith(".in")
    )
    if not names:
        print(f"no .in files in {input_dir}", file=sys.stderr)
        return 1
    for name in names:
        case_id = name[:-3]
        with open(os.path.join(input_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        out_path = os.path.join(output_dir, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh)
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
