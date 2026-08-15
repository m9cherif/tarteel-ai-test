# Territory & Pipeline Forecaster

A quarterly bookings forecaster for a B2B sales organisation. It answers three
questions a VP of Sales actually asks, in order:

1. **Where will the quarter land, and how confident should I be?**
2. **Why should I believe that number?**
3. **What do I do about it this week?**

The output is a distribution, not a point estimate. Bookings are the sum of a
few hundred coin flips of wildly different sizes, so "6.4M" is precise in a way
the underlying reality is not. The forecast is a range and a probability of
clearing quota — which is the question the board is really asking.

```bash
pip install -r requirements.txt
streamlit run app.py
pytest                    # 34 tests
```

---

## What it does

| View | Question it answers |
| --- | --- |
| **Forecast** | Monte Carlo distribution of quarter-end bookings; bottom-up vs top-down |
| **Pipeline** | Coverage by territory and segment; how late deals actually close |
| **Model** | Calibration, discrimination, and what drives a win |
| **Backtest** | How wrong this has been on previous quarters |
| **Coverage** | Where next quarter's selling hours should go, and which deals decide it |
| **Brief** | The paragraph you would open the QBR with |

## How the forecast is built

Three components, because leaving any one out biases the number in a predictable
direction:

- **Committed** — already closed-won this quarter. No longer uncertain.
- **Pipeline** — each open deal is a weighted coin: a calibrated win probability
  times the chance it actually lands before the quarter ends.
- **New business** — deals not yet created that will be sourced *and* closed in
  the remaining weeks. Small in Enterprise, material in SMB. Omitting this is
  the most common reason a bottom-up forecast reads low all quarter and then
  mysteriously catches up in the last fortnight.

A deliberately naive top-down model (seasonal decomposition on quarterly
history) runs alongside it and uses none of the pipeline. When the two agree,
that agreement is real evidence. When they diverge, the gap is the finding.

## Five decisions worth explaining

**Training on as-of snapshots, not final states.** The tempting shortcut is to
train on each closed deal's final stage. That produces a superb AUC and no
predictive value, because at forecast time you don't yet know the final stage.
Every training row here is a *snapshot* — the deal as it appeared at one random
moment during its life, paired with the outcome that followed. Open deals get
the same treatment as of today, so train and serve see the same information.
`features.py` enforces this with an explicit forbidden-column list.

**Calibration before discrimination.** If the model says 70% on a hundred deals,
roughly seventy need to land, or every downstream number inherits the bias. The
classifier is isotonic-calibrated and judged on Brier score and a reliability
curve first, AUC second.

**Measuring against an oracle, not against 1.0.** Deals are decided by a
weighted coin, so no model can reach AUC 1.0. An oracle that knows each deal's
eventual path scores **0.795** on this data; the model, restricted to what was
knowable mid-deal, reaches **0.691** — about **65% of the achievable signal**.
That is a far more honest framing than quoting 0.691 against a perfect score.

**Correlated uncertainty, not independent coin flips.** The first backtest put
only 67% of quarters inside a band that should hold 80% — the interval was too
narrow. Forecasts miss because the *whole quarter* turns out better or worse
than the model thought, not because individual deals surprise independently. A
single shared shift applied to every win probability per simulation fixed it:
coverage now lands at 83% against a nominal 80%.

**Slip is measured, not assumed.** Reps enter optimistic close dates. The
simulation bootstraps from the empirical distribution of *actual close minus
promised close* in closed history — which is lopsided, since deals slip late far
more often than they close early.

## Honest results

From the shipped dataset (seed 7, 10 quarters of history, 294 open deals):

| Metric | Value |
| --- | --- |
| ROC AUC | 0.691 (oracle ceiling 0.795) |
| Brier skill vs base rate | +10% |
| Backtest mean error | 27% across 6 quarters |
| Backtest median error | 19% |
| P10–P90 coverage | 83% (nominal 80%) |

Mean and median error are both reported because a single bad quarter dominates
the mean over six observations, and quoting whichever flatters the model would
be picking the metric after seeing the result. The worst quarter is 2025-Q1,
over-forecast by 69%: at that point the model had only four quarters of history
and could not yet learn that Q1 sources far less same-quarter business than Q4.
Coverage measured on six quarters is itself noisy — it is a directional check,
not a precise calibration.

## Where the hours should go

The allocation model rests on one idea: **attention is worth most where the
outcome is still in doubt.** A deal at 95% closes whether or not anyone works
it; a deal at 5% will not close either way. The value genuinely in play is
`amount × p × (1 − p)` — largest at a coin flip. Effort has diminishing returns,
so the response curve is `swing × (1 − exp(−h/τ))`; because that is concave and
separable, giving each next hour to whichever cell has the highest marginal
return is optimal, not a heuristic.

Two results worth reporting, including the unflattering one:

- **Reallocating hours across territory × segment is worth very little** (~0.3%).
  Pipeline-weighted attention — what any commission plan produces on its own —
  is already close to optimal at that grain. This is a real finding, not a
  failure of the optimiser.
- **The concentration is at deal level.** A handful of coin-flip deals carry a
  large share of the quarter's undecided revenue, and those are what a manager
  can act on this week. The Coverage tab surfaces that list.

An earlier version recommended gutting Enterprise coverage. That was a modelling
error, not an insight: it assumed every deal absorbs the same selling hours, so
cells with few large deals saturated after a few dozen hours. Effort per deal now
varies by segment (SMB 3h, Mid-Market 12h, Enterprise 45h) and the advice is a
modest tilt rather than a retreat.

## The data is synthetic, and deliberately so

There is no real customer data here and there should not be — prospect records
are personal data and belong in a CRM, not a public repository. `src/data.py`
generates a B2B opportunity ledger from an explicit, seeded model of how deals
behave, with the "true" win-probability coefficients written out in the open.

That is the point rather than a limitation: because the real drivers are known,
the modelling code can be judged on whether it **recovers** them. It does — the
model reproduces the true ordering of every driver (source, segment, territory)
with gaps under six points, which is a stronger check than any accuracy number
on data whose ground truth nobody knows.

## The AI layer

The executive brief is written by Claude (`claude-opus-5`) when
`ANTHROPIC_API_KEY` is set. Two constraints:

- **It never invents a figure.** Every number is computed by the model and passed
  in already rounded. Claude writes prose around a supplied fact sheet; it does
  not do arithmetic on a pipeline it cannot see.
- **It is optional.** With no key or no `anthropic` package, the same brief is
  assembled from a template. A forecasting tool should not need a paid API to
  show a forecast.

## Layout

```
src/data.py         synthetic CRM generator; the generative model of winning
src/features.py     as-of snapshots and the leakage guard
src/model.py        calibrated win-probability model + evaluation
src/forecast.py     Monte Carlo, top-down baseline, walk-forward backtest
src/allocation.py   selling-hour allocation and deal triage
src/narrative.py    executive brief (Claude, with a template fallback)
app.py              Streamlit front end
tests/              34 tests
```

## What I would add next

- Rep-level and deal-age cohorts in the win model — currently only rep tenure.
- A proper time-series model (ETS or ARIMA) as a third forecast, instead of the
  seasonal decomposition standing in for one.
- Per-cell effort curves fitted from activity data rather than assumed constants;
  `HOURS_PER_DEAL` is the least evidenced number in the project.
- Longer backtest history so coverage can be calibrated on more than 6 quarters.
