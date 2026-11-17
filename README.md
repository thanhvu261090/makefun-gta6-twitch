# makefun-gta6-twitch

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/thanhvu261090/makefun-gta6-twitch/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/thanhvu261090/makefun-gta6-twitch/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A forecaster for GTA VI's launch-day Twitch concurrent viewers, hour by hour (hours 2–24), as five quantiles per hour. Method: per-hour local-linear kernel regression of log viewers (as ratios to hour 1) on log hour-1 across the past big game launches in the case file — the log-ratio curve is strongly curved in log hour-1 (small hour-1 launches surge by orders of magnitude as the US unlocks, huge ones stay flat), so a local line follows the slope where a local average bends. Plus a pre-release audience-mix correction: the comparables are re-predicted leave-one-out, the residuals are regressed on the pre-release evening ratios (log evening / hour-1 and log evening / overnight), the coefficients are halved, and the correction is capped at ±0.25 log units so a much-anticipated launch never extrapolates the fit past the data. Median and spread are lightly smoothed across neighbouring hours, then turned into quantiles with a normal spread. Deterministic, Python standard library only, no network.

Local validation on the four practice cases: mean pinball loss **0.183** (v1 k-nearest-neighbours: 0.238; house baseline: 0.387), per case 0.197 / 0.192 / 0.144 / 0.196; leave-one-out CV over the 41 past launches: **0.360** (v1: 0.410). Typical miss 22–38% per case, 90% coverage 22–23 of 23 hours.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads one case file per launch from the input directory (`/input/<id>.in`, or `$MAKEFUN_INPUT_DIR`) and writes the forecast for that case to the output directory (`/output/<id>.out`, or `$MAKEFUN_OUTPUT_DIR`) as `{"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}]}` for hours 2..24.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
