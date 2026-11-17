# GTA VI Twitch CCU Forecaster

An analogue-template quantile forecaster for the make.fun bounty
[Forecast GTA VI's launch-day Twitch viewers hour by hour](https://make.fun/bounty/gta6-twitch-ccu-forecast-01).

![competition](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)
![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/bapenven-ctrl/gta6-twitch-forecast/rank.svg)
![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/bapenven-ctrl/gta6-twitch-forecast/reward.svg)

## What it does

`main.py` (Python 3.13, standard library only) reads one case — everything
recorded up to the end of hour 1 of the release day, that hour's Twitch
viewer count included, plus dozens of past big launches' full hourly
curves — and writes, for each of hours 2 to 24, five quantiles (5%, 25%,
50%, 75%, 95%) of that hour's concurrent viewers in the Grand Theft Auto VI
Twitch category.

## Method

The house baseline is naive persistence: every hour equals hour 1. This
entry instead learns the launch day's *shape* from the case's own
comparables:

1. For each comparable launch, form its release-day log-ratio curve
   ln(viewers_h / viewers_1).
2. Keep the rolling-midnight launches — hour 1 already at least 5% of the
   day's peak — the regime GTA VI and all four practice launches are in.
3. Weight each curve by a Gaussian kernel on the distance between its log
   hour-1 audience and the target's, so big launches lend their shape to
   big launches.
4. Forecast hour h as a lognormal anchored on the target's hour-1 print,
   centred and spread by the kernel-weighted mean and standard deviation of
   the pool's log-ratios at h.

Leave-one-out over the practice cases' comparables scores this at a mean
log pinball loss of about 0.254 on rolling-midnight launches, against 0.321
for persistence on the same launches (the baseline's published practice
loss is 0.387). The program is deterministic and runs in well under a
second per case.

## Running it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

It reads every `<id>.in` in the input directory and writes one `<id>.out`
per case: `{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}`
with exactly 23 rows, hours 2 to 24.

## License

MIT — see [LICENSE](LICENSE).
