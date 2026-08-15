"""
As-of feature construction.

The whole file exists to answer one question honestly: *what did we actually
know about this deal at the moment we had to forecast it?*

The tempting shortcut is to train on each closed deal's final state — the stage
it died in, how long it ultimately ran. That produces a model with a superb
AUC and no predictive value whatsoever, because at forecast time you do not yet
know the final stage. It is the single most common way a pipeline model looks
brilliant in a notebook and useless in production.

So every training row is a *snapshot*: the deal as it appeared at one random
moment during its life, paired with the outcome that eventually followed. Open
deals get exactly the same treatment, snapshotted at today. Train and serve see
the same shape of information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Features the model is allowed to see.
CATEGORICAL_FEATURES = ["segment", "territory", "industry", "source"]
NUMERIC_FEATURES = [
    "log_amount",
    "stage_now",
    "touches_now",
    "days_open",
    "days_in_stage",
    "touches_per_week",
    "tenure_months",
    "close_quarter_num",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Columns that exist in the ledger but must never be used as inputs. Some are
# simulation internals; the rest are only knowable once the deal has closed.
# `won` is absent deliberately — it is the label, so it belongs in the training
# frame and is excluded from the model by never appearing in FEATURES.
FORBIDDEN = [
    "cycle_days",
    "final_stage",
    "quality",
    "true_win_p",
    "close_date",
    "close_quarter",
    "is_open",
]


def _observed_state(
    sub: pd.DataFrame, elapsed_days: np.ndarray, rng: np.random.Generator
) -> pd.DataFrame:
    """
    Reconstruct what a CRM would have shown for these deals after `elapsed_days`.

    A deal walks its stages at a roughly even pace across its life, so a deal
    that ends in Negotiation has passed through Qualify, Discovery and Proposal
    on the way. Touches accumulate over the same span.
    """
    cycle = sub.cycle_days.to_numpy()
    final_stage = sub.final_stage.to_numpy()
    elapsed = np.clip(elapsed_days, 0.0, cycle)
    progress = np.divide(elapsed, cycle, out=np.zeros_like(elapsed), where=cycle > 0)

    n_steps = final_stage + 1  # stages this deal will pass through
    stage_now = np.minimum(np.floor(progress * n_steps).astype(int), final_stage)

    # Time spent in the stage the deal is sitting in right now.
    seg_len = cycle / n_steps
    days_in_stage = elapsed - stage_now * seg_len

    # Touches land uniformly over the life, so by `progress` a binomial share
    # of them has happened.
    total = sub.total_touches.to_numpy()
    touches_now = rng.binomial(total, np.clip(progress, 0.0, 1.0))

    weeks_open = np.maximum(elapsed / 7.0, 1.0)

    out = pd.DataFrame(
        {
            "opp_id": sub.opp_id.to_numpy(),
            "segment": sub.segment.to_numpy(),
            "territory": sub.territory.to_numpy(),
            "industry": sub.industry.to_numpy(),
            "source": sub.source.to_numpy(),
            "amount": sub.amount.to_numpy(),
            "log_amount": np.log10(sub.amount.to_numpy()),
            "stage_now": stage_now,
            "touches_now": touches_now,
            "days_open": elapsed,
            "days_in_stage": np.maximum(days_in_stage, 0.0),
            "touches_per_week": touches_now / weeks_open,
            "tenure_months": sub.tenure_months.to_numpy(),
            "close_quarter_num": sub.expected_close_date.dt.quarter.to_numpy(),
            "expected_close_date": sub.expected_close_date.to_numpy(),
            "rep_id": sub.rep_id.to_numpy(),
        }
    )
    return out


def training_frame(
    opps: pd.DataFrame, asof: pd.Timestamp, rng: np.random.Generator
) -> pd.DataFrame:
    """
    Deals that had already closed by `asof`, each seen at one random moment
    during its life, labelled with what eventually happened.

    The snapshot point is drawn between 15% and 95% of the way through: before
    15% almost nothing is known, and after 95% the deal is effectively decided.
    """
    closed = opps[(~opps.is_open) & (opps.close_date <= asof)]
    if closed.empty:
        return pd.DataFrame(columns=FEATURES + ["opp_id", "won"])

    fraction = rng.uniform(0.15, 0.95, size=len(closed))
    elapsed = closed.cycle_days.to_numpy() * fraction

    frame = _observed_state(closed, elapsed, rng)
    frame["won"] = closed.won.astype("boolean").fillna(False).to_numpy().astype(int)
    # Underscore-prefixed: used to split train from test by time, never a feature.
    frame["_close_date"] = closed.close_date.to_numpy()
    # The generator's own win probability, kept only to measure how much of the
    # available signal the model manages to recover. Never an input.
    frame["_true_win_p"] = closed.true_win_p.to_numpy()
    return frame


def open_frame(
    opps: pd.DataFrame, asof: pd.Timestamp, rng: np.random.Generator
) -> pd.DataFrame:
    """
    Deals still running at `asof`, seen as of that date.

    Used both for the live forecast (asof = today) and for backtests, where
    `asof` is rolled back to the start of a historical quarter to rebuild the
    pipeline as it stood then.
    """
    live = opps[(opps.created_date <= asof) & (opps.close_date > asof)]
    if live.empty:
        return pd.DataFrame(columns=FEATURES + ["opp_id", "amount"])

    elapsed = (asof - live.created_date).dt.total_seconds().to_numpy() / 86400.0
    frame = _observed_state(live, elapsed, rng)

    # Kept for scoring a backtest, never fed to the model.
    frame["_eventual_won"] = live.won.astype("boolean").fillna(False).to_numpy()
    frame["_true_close_date"] = live.close_date.to_numpy()
    return frame


def check_no_leakage(frame: pd.DataFrame) -> None:
    """
    Raise if anything the model will see is knowable only after the fact.

    Two ways that can happen, so both are checked: a post-hoc column sneaking
    into FEATURES, and a post-hoc column riding along in the frame where a
    later edit might pick it up.
    """
    in_features = sorted(set(FEATURES) & set(FORBIDDEN))
    if in_features:
        raise ValueError(f"post-hoc columns listed as features: {in_features}")

    carried = [c for c in FORBIDDEN if c in frame.columns]
    if carried:
        raise ValueError(f"post-hoc columns present in model frame: {carried}")
