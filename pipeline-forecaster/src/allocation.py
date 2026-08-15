"""
Where should the team spend next quarter's selling hours?

A forecast that only says "you will land at 6.2M" leaves the manager with
nothing to do. This module turns the same model into a decision: given a fixed
number of selling hours, which territory and segment should get them.

Two ideas do the work.

**Attention belongs where the outcome is still in doubt.** The instinct is to
chase the biggest pipeline, but a deal at 95% will close whether or not anyone
works it, and a deal at 5% will not close either way. The value at stake in a
deal is `amount × p × (1 - p)` — the classic variance term, largest at a coin
flip. Summed over a cell, that is the revenue genuinely in play there.

**Effort has diminishing returns.** The tenth hour on an account is worth less
than the first, so the response curve is `swing × (1 - exp(-h / tau))`. Because
that is concave and separable, handing each next hour to whichever cell has the
highest marginal return is not a heuristic — it is the optimum.

The comparison is against how time is actually spent today: roughly in
proportion to pipeline value, which is the behaviour the incentive structure
produces on its own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Hours needed to work through most of a deal's swing value, by segment. These
# differ by an order of magnitude and the difference matters: a flat rate makes
# Enterprise cells saturate after a few dozen hours, and the optimiser then
# recommends gutting Enterprise coverage — arithmetic no sales leader would or
# should accept. An Enterprise cycle means many stakeholders, a security review
# and a procurement process; an SMB deal is two calls and a signature.
HOURS_PER_DEAL = {"SMB": 3.0, "Mid-Market": 12.0, "Enterprise": 45.0}
DEFAULT_HOURS_PER_DEAL = 10.0

# No territory can be abandoned, however unpromising the arithmetic looks.
# A quarter of starving a region costs relationships that take years to rebuild.
MIN_SHARE_PER_TERRITORY = 0.04


def build_cells(open_deals: pd.DataFrame, win_p: np.ndarray) -> pd.DataFrame:
    """
    Collapse open pipeline into territory × segment cells.

    `swing` is the revenue whose outcome is still genuinely undecided, and it is
    what the allocator maximises. `expected_value` is reported alongside it
    because it is the number people expect to see — and the gap between the two
    orderings is usually the most interesting thing on the screen.
    """
    work = open_deals.copy()
    work["win_p"] = win_p
    work["expected_value"] = work.amount * work.win_p
    work["swing"] = work.amount * work.win_p * (1.0 - work.win_p)

    cells = (
        work.groupby(["territory", "segment"], as_index=False)
        .agg(
            deals=("opp_id", "count"),
            pipeline=("amount", "sum"),
            expected_value=("expected_value", "sum"),
            swing=("swing", "sum"),
            mean_win_p=("win_p", "mean"),
        )
        .sort_values("swing", ascending=False)
        .reset_index(drop=True)
    )
    per_deal = cells.segment.map(HOURS_PER_DEAL).fillna(DEFAULT_HOURS_PER_DEAL)
    cells["hours_per_deal"] = per_deal
    cells["tau"] = np.maximum(cells.deals * per_deal, 1.0)
    return cells


def response(hours: np.ndarray, swing: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Expected incremental bookings from `hours` of selling time in each cell."""
    return swing * (1.0 - np.exp(-hours / tau))


def marginal(hours: np.ndarray, swing: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Derivative of `response` — bookings per additional hour."""
    return (swing / tau) * np.exp(-hours / tau)


def current_allocation(cells: pd.DataFrame, capacity: float) -> np.ndarray:
    """
    The status quo: hours spread in proportion to pipeline value.

    This is not a straw man. Reps work their biggest deals, and pipeline-weighted
    attention is what any commission plan tied to bookings will produce.
    """
    weight = cells.pipeline.to_numpy(dtype=float)
    if weight.sum() <= 0:
        return np.full(len(cells), capacity / max(len(cells), 1))
    return capacity * weight / weight.sum()


def optimise(
    cells: pd.DataFrame,
    capacity: float,
    min_share_per_territory: float = MIN_SHARE_PER_TERRITORY,
    steps: int = 4_000,
) -> np.ndarray:
    """
    Hand out hours one slice at a time, always to the cell that gains most.

    Optimal because the objective is concave and separable: once a cell's
    marginal return drops below another's, it never overtakes again.
    """
    swing = cells.swing.to_numpy(dtype=float)
    tau = cells.tau.to_numpy(dtype=float)
    hours = np.zeros(len(cells), dtype=float)

    # Territory floors are reserved first, split inside the territory by swing.
    territories = cells.territory.to_numpy()
    reserved = 0.0
    for terr in pd.unique(territories):
        mask = territories == terr
        floor = capacity * min_share_per_territory
        share = swing[mask]
        split = share / share.sum() if share.sum() > 0 else np.full(mask.sum(), 1 / mask.sum())
        hours[mask] += floor * split
        reserved += floor

    remaining = max(capacity - reserved, 0.0)
    if remaining <= 0 or swing.sum() <= 0:
        return hours

    slice_size = remaining / steps
    for _ in range(steps):
        gains = marginal(hours, swing, tau)
        hours[int(np.argmax(gains))] += slice_size

    return hours


def triage_deals(
    open_deals: pd.DataFrame, win_p: np.ndarray, top_n: int = 20
) -> pd.DataFrame:
    """
    The individual deals holding the most undecided revenue.

    Reallocating hours between territory × segment cells turns out to be worth
    very little — the status quo is already close to optimal at that grain. The
    concentration is at deal level: a handful of coin-flip deals carry a large
    share of the quarter's uncertainty, and those are what a manager can
    actually act on this week.

    `swing_share` is the fraction of all undecided revenue each deal represents.
    """
    work = open_deals.copy()
    work["win_p"] = win_p
    work["expected_value"] = work.amount * work.win_p
    work["swing"] = work.amount * work.win_p * (1.0 - work.win_p)

    total_swing = work.swing.sum()
    work["swing_share"] = work.swing / total_swing if total_swing > 0 else 0.0

    cols = [
        "opp_id", "territory", "segment", "industry", "source",
        "amount", "win_p", "expected_value", "swing", "swing_share",
        "stage_now", "days_in_stage", "expected_close_date",
    ]
    cols = [c for c in cols if c in work.columns]
    return work.nlargest(top_n, "swing")[cols].reset_index(drop=True)


def allocate(
    open_deals: pd.DataFrame,
    win_p: np.ndarray,
    capacity_hours: float,
    min_share_per_territory: float = MIN_SHARE_PER_TERRITORY,
) -> tuple[pd.DataFrame, dict]:
    """
    Full recommendation: current vs optimised hours per cell, and what the
    reallocation is worth.

    Returns (per-cell table, summary dict).
    """
    cells = build_cells(open_deals, win_p)
    if cells.empty:
        return cells, {"uplift": 0.0, "uplift_pct": 0.0, "capacity_hours": capacity_hours}

    swing = cells.swing.to_numpy(dtype=float)
    tau = cells.tau.to_numpy(dtype=float)

    now = current_allocation(cells, capacity_hours)
    rec = optimise(cells, capacity_hours, min_share_per_territory)

    yield_now = response(now, swing, tau)
    yield_rec = response(rec, swing, tau)

    out = cells.copy()
    out["current_hours"] = now
    out["recommended_hours"] = rec
    out["hours_delta"] = rec - now
    out["yield_now"] = yield_now
    out["yield_recommended"] = yield_rec
    out["uplift"] = yield_rec - yield_now
    out = out.sort_values("uplift", ascending=False).reset_index(drop=True)

    total_now = float(yield_now.sum())
    total_rec = float(yield_rec.sum())
    summary = {
        "capacity_hours": float(capacity_hours),
        "yield_now": total_now,
        "yield_recommended": total_rec,
        "uplift": total_rec - total_now,
        "uplift_pct": (total_rec / total_now - 1.0) if total_now > 0 else 0.0,
        "cells": int(len(out)),
    }
    return out, summary
