"""
Synthetic CRM generator.

There is no real customer data in this project and there never should be —
prospect records are personal data and belong in a CRM, not a public repo.
So we generate a B2B opportunity ledger from an explicit, seeded model of how
deals actually behave, and everything downstream is fitted against that.

The generative model is written out in the open (see WIN_LOGIT_* below) which
means the "true" drivers are known. That is the point: it lets the modelling
code be judged on whether it recovers them, rather than on whether a number
looks plausible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SEED = 7
ANCHOR = pd.Timestamp("2026-08-15")  # fixed so every run is reproducible

TERRITORIES = ["France", "DACH", "Benelux", "Iberia", "Nordics", "UK & Ireland"]
SEGMENTS = ["SMB", "Mid-Market", "Enterprise"]
INDUSTRIES = [
    "Manufacturing",
    "Retail",
    "Financial Services",
    "Healthcare",
    "Public Sector",
    "Technology",
]
SOURCES = ["Inbound", "Outbound", "Partner", "Event", "Referral"]

# Pipeline stages, in order. Index doubles as the numeric stage.
STAGES = ["Qualify", "Discovery", "Proposal", "Negotiation"]

# ---------------------------------------------------------------------------
# The generative model of winning. These are the coefficients the ML model is
# later asked to recover from data alone.
# ---------------------------------------------------------------------------
WIN_LOGIT_INTERCEPT = -2.30

WIN_LOGIT_SOURCE = {
    "Referral": 0.95,
    "Partner": 0.62,
    "Inbound": 0.30,
    "Event": -0.05,
    "Outbound": -0.45,
}

WIN_LOGIT_SEGMENT = {"SMB": 0.35, "Mid-Market": 0.0, "Enterprise": -0.40}

WIN_LOGIT_INDUSTRY = {
    "Technology": 0.28,
    "Financial Services": 0.12,
    "Retail": 0.02,
    "Manufacturing": -0.05,
    "Healthcare": -0.15,
    "Public Sector": -0.35,
}

WIN_LOGIT_TERRITORY = {
    "France": 0.14,
    "DACH": 0.05,
    "Benelux": 0.00,
    "Iberia": -0.10,
    "Nordics": 0.08,
    "UK & Ireland": -0.06,
}

# Reaching a later stage is by far the strongest signal.
WIN_LOGIT_STAGE = {0: -0.85, 1: -0.15, 2: 0.75, 3: 1.65}

SEGMENT_PARAMS = {
    #                 median deal   log-sd   cycle days   deals/quarter
    "SMB":        dict(amount=9_000,   sd=0.55, cycle=34,  volume=210),
    "Mid-Market": dict(amount=48_000,  sd=0.60, cycle=82,  volume=95),
    "Enterprise": dict(amount=190_000, sd=0.70, cycle=155, volume=28),
}

# Q4 pulls deals in; Q1 is slow. Multiplies deal volume.
QUARTER_SEASONALITY = {1: 0.88, 2: 1.02, 3: 0.94, 4: 1.16}

N_REPS = 18
N_HISTORY_QUARTERS = 10

# Quarters generated before the reporting window opens. Without these the first
# reported quarters would under-count closes, because no deal would have been
# created early enough to land in them — an artefact of the window, not a trend.
BURN_IN_QUARTERS = 3


@dataclass
class Config:
    """Knobs for the generator. Defaults produce the shipped dataset."""

    seed: int = SEED
    anchor: pd.Timestamp = ANCHOR
    history_quarters: int = N_HISTORY_QUARTERS
    n_reps: int = N_REPS
    territories: list[str] = field(default_factory=lambda: list(TERRITORIES))


def quarter_start(ts: pd.Timestamp) -> pd.Timestamp:
    """First day of the calendar quarter containing `ts`."""
    return pd.Timestamp(year=ts.year, month=3 * ((ts.month - 1) // 3) + 1, day=1)


def quarter_label(ts: pd.Timestamp) -> str:
    return f"{ts.year}-Q{(ts.month - 1) // 3 + 1}"


def add_quarters(ts: pd.Timestamp, n: int) -> pd.Timestamp:
    """Shift a quarter-start timestamp by n quarters."""
    q = (ts.month - 1) // 3 + n
    return pd.Timestamp(year=ts.year + q // 4, month=3 * (q % 4) + 1, day=1)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def build_reps(rng: np.random.Generator, cfg: Config) -> pd.DataFrame:
    """Sales roster. Tenure matters: ramped reps close better."""
    territories = np.resize(np.array(cfg.territories), cfg.n_reps)
    tenure = rng.integers(2, 96, size=cfg.n_reps)  # months with the company
    return pd.DataFrame(
        {
            "rep_id": [f"R{i:03d}" for i in range(cfg.n_reps)],
            "rep_territory": territories,
            "tenure_months": tenure,
            # Selling capacity is what the allocation model later spends.
            "quarterly_selling_hours": rng.integers(300, 380, size=cfg.n_reps),
        }
    )


def generate_opportunities(cfg: Config | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the opportunity ledger and the rep roster.

    Returns (opportunities, reps). Opportunities span `history_quarters` closed
    quarters plus the open pipeline sitting in the current quarter.
    """
    cfg = cfg or Config()
    rng = np.random.default_rng(cfg.seed)
    reps = build_reps(rng, cfg)

    current_q = quarter_start(cfg.anchor)
    report_start = add_quarters(current_q, -cfg.history_quarters)
    first_q = add_quarters(report_start, -BURN_IN_QUARTERS)
    n_cohorts = cfg.history_quarters + BURN_IN_QUARTERS + 1

    rows: list[dict] = []
    opp_seq = 0

    # One cohort of opportunities created per quarter, including the current one.
    for q_index in range(n_cohorts):
        q_start = add_quarters(first_q, q_index)
        q_num = (q_start.month - 1) // 3 + 1
        # Gentle underlying growth, so a top-down trend model has something real
        # to find rather than pure noise.
        growth = 1.0 + 0.021 * (q_index - BURN_IN_QUARTERS)

        for segment, params in SEGMENT_PARAMS.items():
            n = rng.poisson(params["volume"] * QUARTER_SEASONALITY[q_num] * growth)
            if n == 0:
                continue

            created_offset = rng.integers(0, 90, size=n)
            created = [q_start + pd.Timedelta(days=int(d)) for d in created_offset]

            amounts = rng.lognormal(np.log(params["amount"]), params["sd"], size=n)
            amounts = np.round(amounts / 500) * 500  # deals are quoted in round numbers
            amounts = np.clip(amounts, 1_000, None)

            cycles = np.clip(
                rng.normal(params["cycle"], params["cycle"] * 0.32, size=n), 7, None
            )

            territory = rng.choice(cfg.territories, size=n)
            industry = rng.choice(INDUSTRIES, size=n)
            # Enterprise leans outbound/partner; SMB leans inbound.
            src_p = (
                [0.42, 0.18, 0.13, 0.15, 0.12]
                if segment == "SMB"
                else [0.22, 0.34, 0.20, 0.14, 0.10]
            )
            source = rng.choice(SOURCES, size=n, p=src_p)

            for i in range(n):
                terr = str(territory[i])
                pool = reps[reps.rep_territory == terr]
                if pool.empty:
                    pool = reps
                rep = pool.iloc[int(rng.integers(0, len(pool)))]

                opp_seq += 1
                rows.append(
                    {
                        "opp_id": f"O{opp_seq:06d}",
                        "created_date": created[i],
                        "segment": segment,
                        "territory": terr,
                        "industry": str(industry[i]),
                        "source": str(source[i]),
                        "amount": float(amounts[i]),
                        "cycle_days": float(cycles[i]),
                        "rep_id": rep.rep_id,
                        "tenure_months": int(rep.tenure_months),
                    }
                )

    opps = pd.DataFrame(rows)

    # ---- how far each deal progressed, and whether it landed -------------
    n = len(opps)

    # Touches accumulate over the cycle; more contact genuinely helps, with
    # diminishing returns, and Enterprise needs more of it.
    touch_rate = np.where(opps.segment == "Enterprise", 0.16, 0.11)
    total_touches = np.maximum(
        1, rng.poisson(opps.cycle_days.to_numpy() * touch_rate)
    )

    # Latent quality drives both how far a deal gets and whether it closes,
    # which is what makes stage a strong (but not deterministic) predictor.
    quality = rng.normal(0, 1, size=n)

    stage_reached = np.clip(
        np.round(1.55 + 0.85 * quality + rng.normal(0, 0.55, size=n)), 0, 3
    ).astype(int)

    logit = (
        WIN_LOGIT_INTERCEPT
        + opps.source.map(WIN_LOGIT_SOURCE).to_numpy()
        + opps.segment.map(WIN_LOGIT_SEGMENT).to_numpy()
        + opps.industry.map(WIN_LOGIT_INDUSTRY).to_numpy()
        + opps.territory.map(WIN_LOGIT_TERRITORY).to_numpy()
        + pd.Series(stage_reached).map(WIN_LOGIT_STAGE).to_numpy()
        + 0.55 * np.log1p(total_touches) / np.log(10)
        + 0.010 * np.minimum(opps.tenure_months.to_numpy(), 48)
        + 0.30 * quality
    )
    win_p = _sigmoid(logit)
    won = rng.random(n) < win_p

    opps["total_touches"] = total_touches
    opps["final_stage"] = stage_reached
    opps["quality"] = quality
    opps["true_win_p"] = win_p
    # Nullable boolean: an open deal's outcome is genuinely unknown, not False.
    opps["won"] = pd.array(won, dtype="boolean")

    close_date = opps.created_date + pd.to_timedelta(opps.cycle_days, unit="D")
    opps["close_date"] = close_date

    # What the rep typed into the CRM. Reps are systematically optimistic about
    # close dates, so this runs ~12% short of reality with noise on top. The
    # forecaster is only allowed to see this field, never the true close date —
    # modelling the slip between the two is half the job.
    optimism = rng.normal(0.88, 0.16, size=n).clip(0.45, 1.6)
    opps["expected_close_date"] = opps.created_date + pd.to_timedelta(
        opps.cycle_days * optimism, unit="D"
    )

    # Anything still running at the anchor date is open pipeline.
    opps["is_open"] = opps.close_date > cfg.anchor
    opps.loc[opps.is_open, "won"] = pd.NA

    # Deals created after the anchor have not been sourced yet — a CRM on this
    # date would not contain them. Leaving them in inflates open pipeline with
    # opportunities nobody has found, and the ones a backtest legitimately needs
    # (created after a past quarter opened, but before today) are still here.
    opps = opps[opps.created_date <= cfg.anchor].reset_index(drop=True)

    opps["created_quarter"] = opps.created_date.map(quarter_label)
    opps["close_quarter"] = opps.close_date.map(quarter_label)

    # Burn-in cohorts exist only so the reported quarters are fully populated;
    # they are not part of the analysable history.
    opps["in_window"] = opps.close_date >= report_start
    opps.attrs["report_start"] = report_start
    opps.attrs["current_quarter"] = quarter_label(current_q)
    opps.attrs["anchor"] = cfg.anchor

    return opps, reps


def actuals_by_quarter(opps: pd.DataFrame, complete_only: bool = True) -> pd.DataFrame:
    """
    Closed-won bookings per quarter — the series a top-down model forecasts.

    The current quarter is still in flight, so by default it is excluded: a
    part-finished quarter compared against finished ones would read as a
    collapse in bookings rather than as a quarter that has not ended yet.
    """
    closed = opps[(~opps.is_open) & opps.in_window].copy()
    booked = (
        closed[closed.won.astype("boolean").fillna(False)]
        .groupby("close_quarter", as_index=False)
        .agg(bookings=("amount", "sum"), deals=("opp_id", "count"))
        .sort_values("close_quarter")
        .reset_index(drop=True)
    )
    if complete_only:
        booked = booked[booked.close_quarter != opps.attrs["current_quarter"]]
    return booked.reset_index(drop=True)
