"""
Quarter forecasting, bottom-up and top-down.

A single number is the wrong answer to "where will we land?". Bookings are the
sum of a few hundred coin flips of wildly different sizes, so the honest output
is a distribution: a likely range, and the probability of clearing quota. A
board asking "are we going to make it?" is asking about that probability, not
about a point estimate that will certainly be wrong.

Three things go into the quarter:

  committed    already closed-won since the quarter opened — no longer uncertain
  pipeline     open deals, each a weighted coin: win probability × the chance it
               actually lands before the quarter ends
  new business deals not yet created that will be sourced and closed inside the
               remaining weeks — small in Enterprise, material in SMB

Ignoring the third is the most common way a bottom-up forecast reads low all
quarter and then mysteriously catches up in the last fortnight.

The top-down model is deliberately naive — seasonal decomposition on quarterly
history — and exists to disagree. When bottom-up and top-down diverge sharply,
that gap is the finding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import add_quarters, quarter_label, quarter_start

N_SIMULATIONS = 20_000

# Spread of the shared per-quarter shift applied to every win probability, in
# logits. Tuned so backtested P10–P90 coverage lands near its nominal 80%.
CONDITION_SIGMA = 0.35


@dataclass
class ForecastResult:
    """A distribution over quarter-end bookings, plus its parts."""

    quarter: str
    simulations: np.ndarray
    committed: float
    pipeline_value: float
    quota: float

    @property
    def p10(self) -> float:
        return float(np.percentile(self.simulations, 10))

    @property
    def p50(self) -> float:
        return float(np.percentile(self.simulations, 50))

    @property
    def p90(self) -> float:
        return float(np.percentile(self.simulations, 90))

    @property
    def mean(self) -> float:
        return float(self.simulations.mean())

    def probability_of_quota(self) -> float:
        if self.quota <= 0:
            return float("nan")
        return float((self.simulations >= self.quota).mean())

    def summary(self) -> str:
        return (
            f"{self.quarter}: P50 {self.p50/1e6:.2f}M "
            f"(P10 {self.p10/1e6:.2f}M – P90 {self.p90/1e6:.2f}M), "
            f"{self.probability_of_quota():.0%} chance of quota"
        )


def quarter_end(q_start: pd.Timestamp) -> pd.Timestamp:
    """Last moment of the quarter beginning at `q_start`."""
    return add_quarters(q_start, 1) - pd.Timedelta(seconds=1)


def estimate_slip(opps: pd.DataFrame, asof: pd.Timestamp) -> np.ndarray:
    """
    How late deals actually close, in days, versus the date the rep promised.

    Measured from closed history rather than assumed, because slip is exactly
    the kind of thing every sales org believes it has under control. Returns the
    empirical distribution, which the simulation then bootstraps from — its
    shape is lopsided (deals slip late far more than they close early) and a
    symmetric assumption would misprice the end of the quarter.
    """
    closed = opps[(~opps.is_open) & (opps.close_date <= asof)]
    if closed.empty:
        return np.zeros(1)
    slip = (closed.close_date - closed.expected_close_date).dt.total_seconds() / 86400.0
    return slip.to_numpy()


def estimate_new_business(
    opps: pd.DataFrame, asof: pd.Timestamp, q_start: pd.Timestamp
) -> tuple[float, float]:
    """
    Bookings still to come from deals that do not exist yet.

    Estimated from history: in each completed quarter, how much closed-won
    revenue came from deals *created inside that same quarter*. Scaled down by
    how much of the current quarter is already gone, since a deal sourced in the
    last two weeks rarely closes in them.

    Returns (mean, standard deviation) in currency.
    """
    closed = opps[(~opps.is_open) & (opps.close_date <= asof)].copy()
    if closed.empty:
        return 0.0, 0.0

    same_quarter = closed[closed.created_quarter == closed.close_quarter]
    won = same_quarter[same_quarter.won.astype("boolean").fillna(False)]
    if won.empty:
        return 0.0, 0.0

    per_quarter = won.groupby("close_quarter").amount.sum()
    # Drop the quarter in progress: it is partial by definition.
    per_quarter = per_quarter[per_quarter.index != quarter_label(q_start)]
    if per_quarter.empty:
        return 0.0, 0.0

    # Same-quarter sourcing is strongly seasonal — Q1 starts cold, Q4 pulls
    # everything forward — so prefer the history of *this* quarter number.
    # A flat average across all quarters over-forecast Q1 by 68% in backtest.
    target_q = (q_start.month - 1) // 3 + 1
    same_q = per_quarter[per_quarter.index.str[-1].astype(int) == target_q]
    reference = same_q if len(same_q) >= 2 else per_quarter

    q_end = quarter_end(q_start)
    total_days = (q_end - q_start).total_seconds() / 86400.0
    remaining = max((q_end - asof).total_seconds() / 86400.0, 0.0)
    fraction = np.clip(remaining / total_days, 0.0, 1.0)

    sd = reference.std()
    if not np.isfinite(sd):
        sd = per_quarter.std()
    return float(reference.mean() * fraction), float((sd if np.isfinite(sd) else 0.0) * fraction)


def simulate_quarter(
    open_deals: pd.DataFrame,
    win_p: np.ndarray,
    opps: pd.DataFrame,
    asof: pd.Timestamp,
    quota: float = 0.0,
    n_sims: int = N_SIMULATIONS,
    rng: np.random.Generator | None = None,
    include_new_business: bool = True,
    condition_sigma: float = CONDITION_SIGMA,
) -> ForecastResult:
    """
    Monte Carlo the quarter.

    Each simulation flips every open deal against its calibrated win
    probability, draws a slip from the empirical distribution to decide whether
    it lands inside the quarter at all, adds what is already committed, and adds
    a draw of not-yet-created business.

    Every simulation also gets one shared shift applied to all win probabilities
    (`condition_sigma`). Without it the deal-level coin flips are independent,
    the errors cancel across a few hundred deals, and the interval comes out far
    too narrow — the first backtest put only 67% of quarters inside a band that
    should hold 80%. Forecasts miss because the whole quarter turns out better
    or worse than the model thought, not because individual deals surprise
    independently, and that is correlated error.
    """
    rng = rng or np.random.default_rng(7)
    q_start = quarter_start(asof)
    q_end = quarter_end(q_start)

    # Already in the bag this quarter.
    booked = opps[
        (~opps.is_open)
        & (opps.close_date >= q_start)
        & (opps.close_date <= asof)
        & opps.won.astype("boolean").fillna(False)
    ]
    committed = float(booked.amount.sum())

    n_deals = len(open_deals)
    if n_deals == 0:
        sims = np.full(n_sims, committed)
        return ForecastResult(quarter_label(q_start), sims, committed, 0.0, quota)

    amounts = open_deals.amount.to_numpy(dtype=float)
    expected = pd.to_datetime(open_deals.expected_close_date)
    days_to_qend = ((q_end - expected).dt.total_seconds() / 86400.0).to_numpy()

    slip_pool = estimate_slip(opps, asof)

    # (n_sims, n_deals) is the natural shape but explodes on memory for large
    # pipelines, so simulate in chunks and keep only the per-simulation totals.
    totals = np.empty(n_sims, dtype=float)
    chunk = max(1, min(n_sims, int(4e6 // max(n_deals, 1))))

    base_logit = np.log(np.clip(win_p, 1e-6, 1 - 1e-6) / (1 - np.clip(win_p, 1e-6, 1 - 1e-6)))

    for lo in range(0, n_sims, chunk):
        hi = min(lo + chunk, n_sims)
        k = hi - lo
        # One shared condition per simulated quarter, applied to every deal.
        shift = rng.normal(0.0, condition_sigma, size=(k, 1))
        p_sim = 1.0 / (1.0 + np.exp(-(base_logit[None, :] + shift)))
        wins = rng.random((k, n_deals)) < p_sim
        slips = rng.choice(slip_pool, size=(k, n_deals), replace=True)
        lands_in_quarter = slips <= days_to_qend
        totals[lo:hi] = (wins & lands_in_quarter) @ amounts

    if include_new_business:
        nb_mean, nb_sd = estimate_new_business(opps, asof, q_start)
        if nb_mean > 0:
            totals += np.clip(rng.normal(nb_mean, max(nb_sd, 1.0), size=n_sims), 0, None)

    totals += committed

    return ForecastResult(
        quarter=quarter_label(q_start),
        simulations=totals,
        committed=committed,
        pipeline_value=float(amounts.sum()),
        quota=quota,
    )


# ---------------------------------------------------------------------------
# Top-down: seasonal decomposition on the quarterly series
# ---------------------------------------------------------------------------
def topdown_forecast(actuals: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """
    Classical decomposition: strip seasonality, fit a straight line, put the
    seasonality back.

    Not sophisticated, and that is the point — it uses none of the pipeline, so
    when it agrees with the bottom-up number that agreement is real evidence,
    and when it does not, someone should find out why.
    """
    if len(actuals) < 4:
        return pd.DataFrame(columns=["quarter", "forecast"])

    y = actuals.bookings.to_numpy(dtype=float)
    q_num = actuals.close_quarter.str[-1].astype(int).to_numpy()

    overall = y.mean()
    seasonal = {}
    for q in range(1, 5):
        mask = q_num == q
        seasonal[q] = (y[mask].mean() / overall) if mask.any() else 1.0

    deseasonalised = y / np.array([seasonal[q] for q in q_num])
    t = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(t, deseasonalised, 1)

    last_label = actuals.close_quarter.iloc[-1]
    last_start = pd.Timestamp(
        year=int(last_label[:4]), month=3 * (int(last_label[-1]) - 1) + 1, day=1
    )

    rows = []
    for h in range(1, periods + 1):
        nxt = add_quarters(last_start, h)
        q = (nxt.month - 1) // 3 + 1
        trend = intercept + slope * (len(y) - 1 + h)
        rows.append({"quarter": quarter_label(nxt), "forecast": float(trend * seasonal[q])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Walk-forward backtest
# ---------------------------------------------------------------------------
def backtest(
    opps: pd.DataFrame,
    train_fn,
    n_quarters: int = 5,
    n_sims: int = 4_000,
    seed: int = 7,
) -> pd.DataFrame:
    """
    Re-run the whole forecast at the start of each of the last `n_quarters`
    completed quarters, using only what was knowable on that date, and compare
    against what the quarter actually delivered.

    This is the only number in the project that says whether any of it works.
    Coverage matters as much as error: if the P10–P90 band is honest, roughly
    80% of quarters should land inside it.
    """
    from .features import open_frame, training_frame
    from .model import score

    rng = np.random.default_rng(seed)
    current_q = quarter_start(opps.attrs["anchor"])

    rows = []
    for back in range(n_quarters, 0, -1):
        q_start = add_quarters(current_q, -back)
        q_end = quarter_end(q_start)

        train = training_frame(opps, q_start, rng)
        if len(train) < 300:
            continue  # not enough closed history yet to fit anything honest

        model, _ = train_fn(train)
        live = open_frame(opps, q_start, rng)
        if live.empty:
            continue

        p = score(model, live)
        result = simulate_quarter(
            live, p, opps, asof=q_start, n_sims=n_sims, rng=rng
        )

        actual = float(
            opps[
                (~opps.is_open)
                & (opps.close_date >= q_start)
                & (opps.close_date <= q_end)
                & opps.won.astype("boolean").fillna(False)
            ].amount.sum()
        )

        rows.append(
            {
                "quarter": quarter_label(q_start),
                "actual": actual,
                "p10": result.p10,
                "p50": result.p50,
                "p90": result.p90,
                "error": result.p50 - actual,
                "ape": abs(result.p50 - actual) / actual if actual else np.nan,
                "in_band": bool(result.p10 <= actual <= result.p90),
            }
        )

    return pd.DataFrame(rows)


def backtest_summary(bt: pd.DataFrame) -> dict:
    """MAPE and interval coverage across backtested quarters."""
    if bt.empty:
        return {"mape": float("nan"), "coverage": float("nan"), "quarters": 0}
    return {
        # Mean and median both reported: with a handful of quarters a single bad
        # one dominates the mean, and quoting only whichever flatters the model
        # would be picking the metric after seeing the result.
        "mape": float(bt.ape.mean()),
        "median_ape": float(bt.ape.median()),
        "coverage": float(bt.in_band.mean()),
        "quarters": int(len(bt)),
        "bias": float(bt.error.mean()),
    }
