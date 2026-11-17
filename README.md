# makefun-gta6-twitch

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/thanhvu261090/makefun-gta6-twitch/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/thanhvu261090/makefun-gta6-twitch/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A forecaster for GTA VI's launch-day Twitch concurrent viewers, hour by hour (hours 2–24), as five quantiles per hour. Method: k-nearest-neighbour curves from the 42 past big game launches in the case file, matched on the launch's hour-1 viewer print; per-hour empirical quantiles of the neighbours' log-ratios to hour 1, with the spread widened slightly. Big hour-1 prints decay through the day while small ones surge as the US unlocks, so matching on hour 1 captures the unlock-time shape. Deterministic, Python standard library only, no network.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads one case file per launch from the input directory (`/input/<id>.in`, or `$MAKEFUN_INPUT_DIR`) and writes the forecast for that case to the output directory (`/output/<id>.out`, or `$MAKEFUN_OUTPUT_DIR`) as `{"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}]}` for hours 2..24.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
