"""
Tests for the forecasting pipeline.

Weighted toward the things that would be embarrassing to get wrong and easy not
to notice: leakage into the feature set, a Monte Carlo that quietly ignores
committed revenue, and an "optimal" allocator that loses to an equal split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.allocation import allocate, build_cells, optimise, response
from src.data import ANCHOR, Config, actuals_by_quarter, generate_opportunities
from src.features import (
    FEATURES,
    FORBIDDEN,
    check_no_leakage,
    open_frame,
    training_frame,
)
from src.forecast import (
    estimate_slip,
    quarter_end,
    simulate_quarter,
    topdown_forecast,
)
from src.model import score, temporal_split, train_win_model
from src.narrative import generate_brief


@pytest.fixture(scope="module")
def ledger():
    return generate_opportunities()


@pytest.fixture(scope="module")
def frames(ledger):
    opps, _ = ledger
    rng = np.random.default_rng(0)
    return training_frame(opps, ANCHOR, rng), open_frame(opps, ANCHOR, rng)


@pytest.fixture(scope="module")
def fitted(frames):
    train, _ = frames
    model, report = train_win_model(train, fast=True)
    return model, report


# ---------------------------------------------------------------- data ----
def test_generator_is_reproducible():
    a, _ = generate_opportunities(Config(seed=42))
    b, _ = generate_opportunities(Config(seed=42))
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_differ():
    a, _ = generate_opportunities(Config(seed=1))
    b, _ = generate_opportunities(Config(seed=2))
    assert len(a) != len(b) or not a.amount.equals(b.amount)


def test_win_rate_is_plausible_for_b2b(ledger):
    opps, _ = ledger
    closed = opps[(~opps.is_open) & opps.in_window]
    assert 0.20 < closed.won.astype("boolean").mean() < 0.45


def test_enterprise_wins_less_than_smb(ledger):
    """The generator's segment coefficients should show up in the outcomes."""
    opps, _ = ledger
    closed = opps[(~opps.is_open) & opps.in_window]
    by_segment = closed.groupby("segment")["won"].mean()
    assert by_segment["Enterprise"] < by_segment["SMB"]


def test_open_deals_have_not_closed_yet(ledger):
    opps, _ = ledger
    assert (opps[opps.is_open].close_date > ANCHOR).all()
    assert opps[opps.is_open].won.isna().all()


def test_actuals_exclude_the_quarter_in_progress(ledger):
    """A part-finished quarter next to finished ones reads as a collapse."""
    opps, _ = ledger
    actuals = actuals_by_quarter(opps)
    assert opps.attrs["current_quarter"] not in set(actuals.close_quarter)
    assert (actuals.bookings > 0).all()


def test_reps_expect_deals_to_close_earlier_than_they_do(ledger):
    """Rep-entered close dates are optimistic; the forecaster must model that."""
    opps, _ = ledger
    closed = opps[~opps.is_open]
    slip = (closed.close_date - closed.expected_close_date).dt.days
    assert slip.median() > 0


# ------------------------------------------------------------ features ----
def test_no_forbidden_column_is_a_feature():
    assert set(FEATURES).isdisjoint(FORBIDDEN)


def test_leakage_guard_catches_a_smuggled_column(frames):
    train, _ = frames
    sneaky = train.copy()
    sneaky["cycle_days"] = 1.0
    with pytest.raises(ValueError, match="post-hoc"):
        check_no_leakage(sneaky)


def test_snapshot_state_never_runs_ahead_of_the_deal(ledger, frames):
    """
    A snapshot cannot show more progress than the deal ever made — no stage
    beyond its final one, no touches it never received.
    """
    opps, _ = ledger
    train, _ = frames
    merged = train.merge(
        opps[["opp_id", "final_stage", "total_touches"]], on="opp_id", validate="1:1"
    )
    assert (merged.stage_now <= merged.final_stage).all()
    assert (merged.touches_now <= merged.total_touches).all()
    assert (merged.days_in_stage >= 0).all()


def test_open_frame_matches_the_open_pipeline(ledger, frames):
    opps, _ = ledger
    _, live = frames
    assert set(live.opp_id) == set(opps[opps.is_open].opp_id)


def test_training_frame_only_contains_closed_deals(ledger, frames):
    opps, _ = ledger
    train, _ = frames
    assert set(train.opp_id).isdisjoint(set(opps[opps.is_open].opp_id))


# --------------------------------------------------------------- model ----
def test_train_test_split_is_chronological(frames):
    train_frame, _ = frames
    train, test = temporal_split(train_frame)
    assert train._close_date.max() <= test._close_date.min()


def test_model_beats_the_base_rate(fitted):
    _, report = fitted
    assert report.roc_auc > 0.6
    assert report.brier_skill > 0.03


def test_model_does_not_beat_the_oracle(fitted):
    """A model outscoring an oracle that saw the future means leakage."""
    _, report = fitted
    assert report.roc_auc < report.oracle_roc_auc


def test_predictions_are_calibrated(fitted):
    """Averaged over the test set, predicted and observed rates should agree."""
    _, report = fitted
    cal = report.calibration
    weighted_pred = (cal.predicted * cal.deals).sum() / cal.deals.sum()
    weighted_obs = (cal.observed * cal.deals).sum() / cal.deals.sum()
    assert abs(weighted_pred - weighted_obs) < 0.05


def test_stage_is_the_strongest_feature(fitted):
    """The generator makes stage dominant; the model should recover that."""
    _, report = fitted
    assert report.importance.iloc[0].feature == "stage_now"


def test_scores_are_probabilities(fitted, frames):
    model, _ = fitted
    _, live = frames
    p = score(model, live)
    assert len(p) == len(live)
    assert ((p >= 0) & (p <= 1)).all()


# ------------------------------------------------------------ forecast ----
@pytest.fixture(scope="module")
def result(ledger, fitted, frames):
    opps, _ = ledger
    model, _ = fitted
    _, live = frames
    p = score(model, live)
    return simulate_quarter(
        live, p, opps, ANCHOR, quota=6e6, n_sims=3000,
        rng=np.random.default_rng(5),
    )


def test_percentiles_are_ordered(result):
    assert result.p10 < result.p50 < result.p90


def test_forecast_includes_what_is_already_booked(result):
    """Even the pessimistic case cannot fall below revenue already closed."""
    assert result.committed > 0
    assert result.simulations.min() >= result.committed - 1e-6


def test_forecast_cannot_exceed_everything_winning(result):
    ceiling = result.committed + result.pipeline_value
    # New business is added on top, so allow headroom for it.
    assert result.p90 < ceiling * 1.5


def test_quota_probability_is_a_probability(result):
    assert 0.0 <= result.probability_of_quota() <= 1.0


def test_empty_pipeline_forecasts_only_what_is_committed(ledger):
    opps, _ = ledger
    empty = pd.DataFrame(columns=["opp_id", "amount", "expected_close_date"])
    res = simulate_quarter(empty, np.array([]), opps, ANCHOR, n_sims=100)
    assert res.p50 == res.committed


def test_correlated_uncertainty_widens_the_band(ledger, fitted, frames):
    """
    Independent coin flips cancel out across hundreds of deals. The shared
    per-quarter shift is what makes the interval honest, so it must widen it.
    """
    opps, _ = ledger
    model, _ = fitted
    _, live = frames
    p = score(model, live)

    narrow = simulate_quarter(
        live, p, opps, ANCHOR, n_sims=3000, condition_sigma=0.0,
        rng=np.random.default_rng(1), include_new_business=False,
    )
    wide = simulate_quarter(
        live, p, opps, ANCHOR, n_sims=3000, condition_sigma=0.35,
        rng=np.random.default_rng(1), include_new_business=False,
    )
    assert (wide.p90 - wide.p10) > (narrow.p90 - narrow.p10)


def test_slip_distribution_is_right_skewed(ledger):
    """Deals slip late far more than they close early."""
    opps, _ = ledger
    slip = estimate_slip(opps, ANCHOR)
    assert np.percentile(slip, 90) > abs(np.percentile(slip, 10))


def test_topdown_forecasts_the_next_quarter(ledger):
    opps, _ = ledger
    actuals = actuals_by_quarter(opps)
    fc = topdown_forecast(actuals, periods=1)
    assert len(fc) == 1
    assert fc.iloc[0].quarter == opps.attrs["current_quarter"]
    assert fc.iloc[0].forecast > 0


def test_quarter_end_is_the_last_moment_of_the_quarter():
    end = quarter_end(pd.Timestamp("2026-07-01"))
    assert end.year == 2026 and end.month == 9 and end.day == 30


# ---------------------------------------------------------- allocation ----
@pytest.fixture(scope="module")
def cells(fitted, frames):
    model, _ = fitted
    _, live = frames
    return build_cells(live, score(model, live))


def test_swing_is_never_larger_than_expected_value(cells):
    """amount*p*(1-p) <= amount*p for any probability."""
    assert (cells.swing <= cells.expected_value + 1e-9).all()


def test_enterprise_absorbs_more_hours_per_deal(cells):
    """A flat rate here is what produced the 'abandon Enterprise' advice."""
    per_deal = cells.set_index("segment").hours_per_deal.groupby("segment").first()
    assert per_deal["Enterprise"] > per_deal["Mid-Market"] > per_deal["SMB"]


def test_allocation_spends_exactly_the_capacity(cells):
    capacity = 5000.0
    hours = optimise(cells, capacity)
    assert hours.sum() == pytest.approx(capacity, rel=1e-6)
    assert (hours >= 0).all()


def test_every_territory_keeps_a_floor(cells):
    """No quarter should starve a region, however the arithmetic looks."""
    capacity = 5000.0
    hours = optimise(cells, capacity, min_share_per_territory=0.04)
    per_territory = pd.Series(hours).groupby(cells.territory.to_numpy()).sum()
    assert (per_territory >= capacity * 0.04 * 0.999).all()


def test_greedy_allocation_beats_the_alternatives(cells):
    """
    The objective is concave and separable, so marginal-value greedy is optimal.
    Verify it against both baselines it claims to beat.
    """
    capacity = 5000.0
    swing = cells.swing.to_numpy()
    tau = cells.tau.to_numpy()

    greedy = response(optimise(cells, capacity, min_share_per_territory=0.0), swing, tau).sum()
    equal = response(np.full(len(cells), capacity / len(cells)), swing, tau).sum()
    by_pipeline = response(
        capacity * cells.pipeline.to_numpy() / cells.pipeline.sum(), swing, tau
    ).sum()

    assert greedy >= equal
    assert greedy >= by_pipeline


def test_allocation_reports_a_non_negative_uplift(fitted, frames):
    model, _ = fitted
    _, live = frames
    table, summary = allocate(live, score(model, live), capacity_hours=6000.0)
    assert summary["uplift"] >= 0
    assert len(table) == summary["cells"]


# ----------------------------------------------------------- narrative ----
def test_template_brief_is_written_without_an_api_key():
    ctx = {
        "quarter": "2026-Q3", "p50": 6.2e6, "p10": 5.1e6, "p90": 7.4e6,
        "topdown": 5.7e6, "committed": 2.2e6, "pipeline": 24.3e6,
        "n_deals": 294, "quota": 6.0e6, "p_quota": 0.62,
        "mape": 0.24, "coverage": 0.83, "bt_quarters": 6,
        "top_deals_share": 0.41,
    }
    brief = generate_brief(ctx, use_claude=False)
    assert brief.source == "template"
    assert "nan" not in brief.text.lower()
    assert "2026-Q3" in brief.text
    assert len(brief.text.split()) > 60
