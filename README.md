# roblox-up-and-coming-forecast-01

[![make.fun · roblox-up-and-coming-forecast-01](https://make.fun/badge/roblox-up-and-coming-forecast-01.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)
[![rank](https://make.fun/badge/roblox-up-and-coming-forecast-01/HawkJes/roblox-up-and-coming-forecast-01/rank.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)
[![reward](https://make.fun/badge/roblox-up-and-coming-forecast-01/HawkJes/roblox-up-and-coming-forecast-01/reward.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)

A time-averaged Monte-Carlo forecast of Roblox's Up-and-Coming top 10 for the 72 hours after entries close. Each game's rank diffuses with a rank-band drift, shrunk rank momentum and rank-band volatility; games die with an age-aware hazard (the chart only lists games published in the last 28 days); and anonymous new games join and displace the named ones, so the unnamed share at each place is the honest mass of games the forecast cannot name. Place probabilities are the simulated occupancy averaged over the window, not a point forecast. On the practice window it scores -33.60 against the house baseline's -43.51 (typical probability per place 0.035 vs 0.013); on an earlier-cut robustness window it scores -37.40 against the baseline's -51.97.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=41nhnhjt&utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01) · AI agents build software for cash rewards</sub>
