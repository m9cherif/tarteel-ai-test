"""
Territory & Pipeline Forecaster — Streamlit front end.

Six views, in the order a sales leader actually asks the questions:

  Forecast   where does the quarter land, and how sure are we?
  Pipeline   what is the shape of the business underneath that number?
  Model      why should anyone believe the probabilities?
  Backtest   how wrong has this been before?
  Coverage   where should the team spend next quarter's hours?
  Brief      the paragraph you would open the QBR with.

Everything expensive is cached, so moving the quota slider re-runs the Monte
Carlo but not the model fit.
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.allocation import allocate, triage_deals
from src.data import ANCHOR, actuals_by_quarter, generate_opportunities, quarter_start
from src.features import open_frame, training_frame
from src.forecast import (
    backtest,
    backtest_summary,
    estimate_slip,
    simulate_quarter,
    topdown_forecast,
)
from src.model import group_effects, score, train_win_model
from src.narrative import generate_brief

st.set_page_config(
    page_title="Territory & Pipeline Forecaster",
    page_icon="📈",
    layout="wide",
)

ACCENT = "#5F8340"
WARN = "#C97B90"
MUTED = "#8A9480"


# ---------------------------------------------------------------------------
# Cached computation
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Building the opportunity ledger…")
def load_ledger():
    return generate_opportunities()


@st.cache_resource(show_spinner="Fitting the win-probability model…")
def fit_model(seed: int):
    opps, reps = load_ledger()
    rng = np.random.default_rng(seed)
    train = training_frame(opps, ANCHOR, rng)
    model, report = train_win_model(train)
    return model, report, train


@st.cache_data(show_spinner="Scoring open pipeline…")
def score_pipeline(seed: int):
    opps, _ = load_ledger()
    model, _, _ = fit_model(seed)
    rng = np.random.default_rng(seed + 1)
    live = open_frame(opps, ANCHOR, rng)
    live = live.assign(win_p=score(model, live))
    return live


@st.cache_data(show_spinner="Running the walk-forward backtest…")
def run_backtest(seed: int, n_quarters: int):
    opps, _ = load_ledger()
    bt = backtest(
        opps, lambda f: train_win_model(f, fast=True),
        n_quarters=n_quarters, seed=seed,
    )
    return bt, backtest_summary(bt)


def money(x: float) -> str:
    return f"{x/1e6:.2f}M"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
opps, reps = load_ledger()
current_q = opps.attrs["current_quarter"]

st.sidebar.title("Territory & Pipeline Forecaster")
st.sidebar.caption(
    f"Quarter **{current_q}**, as of {ANCHOR:%d %b %Y}. "
    "Synthetic CRM data — see the README."
)

quota = st.sidebar.number_input(
    "Quarterly quota (M)", min_value=1.0, max_value=30.0, value=6.0, step=0.25,
) * 1e6

n_sims = st.sidebar.select_slider(
    "Monte Carlo runs", options=[2_000, 5_000, 10_000, 20_000, 50_000], value=20_000,
)

seed = st.sidebar.number_input("Random seed", min_value=0, max_value=9999, value=7)

capacity_default = float(reps.quarterly_selling_hours.sum())
capacity = st.sidebar.number_input(
    "Selling hours available", min_value=500.0, max_value=20_000.0,
    value=capacity_default, step=100.0,
)

bt_quarters = st.sidebar.slider("Backtest quarters", 3, 8, 6)
use_claude = st.sidebar.checkbox(
    "Write the brief with Claude", value=False,
    help="Needs ANTHROPIC_API_KEY. Without it the brief is written from a template.",
)

# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
model, report, train = fit_model(seed)
live = score_pipeline(seed)
actuals = actuals_by_quarter(opps)

result = simulate_quarter(
    live, live.win_p.to_numpy(), opps, ANCHOR,
    quota=quota, n_sims=int(n_sims), rng=np.random.default_rng(seed + 2),
)
td = topdown_forecast(actuals, periods=1)
topdown_value = float(td.iloc[0].forecast) if len(td) else float("nan")

bt, bt_stats = run_backtest(seed, bt_quarters)
alloc_table, alloc_summary = allocate(live, live.win_p.to_numpy(), capacity)
triage = triage_deals(live, live.win_p.to_numpy(), top_n=20)

tabs = st.tabs(
    ["Forecast", "Pipeline", "Model", "Backtest", "Coverage", "Brief"]
)

# ---------------------------------------------------------------- Forecast --
with tabs[0]:
    st.subheader(f"{current_q} forecast")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Forecast (P50)", money(result.p50), f"{money(result.p50 - quota)} vs quota")
    c2.metric("80% range", f"{money(result.p10)} – {money(result.p90)}")
    c3.metric("Chance of quota", f"{result.probability_of_quota():.0%}")
    c4.metric("Committed so far", money(result.committed))

    st.caption(
        "The range is the forecast. The midpoint is the single number people "
        "quote, and the single number that is always wrong."
    )

    sims = pd.DataFrame({"bookings": result.simulations / 1e6})
    hist = (
        alt.Chart(sims)
        .mark_bar(opacity=0.85, color=ACCENT)
        .encode(
            alt.X("bookings:Q", bin=alt.Bin(maxbins=60), title="Quarter bookings (M)"),
            alt.Y("count()", title="Simulations"),
        )
    )
    marks = pd.DataFrame({
        "value": [result.p10 / 1e6, result.p50 / 1e6, result.p90 / 1e6, quota / 1e6],
        "label": ["P10", "P50", "P90", "Quota"],
    })
    rules = (
        alt.Chart(marks)
        .mark_rule(strokeDash=[5, 4], size=2)
        .encode(
            x="value:Q",
            color=alt.Color(
                "label:N",
                scale=alt.Scale(
                    domain=["P10", "P50", "P90", "Quota"],
                    range=[MUTED, "#1B2417", MUTED, WARN],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=["label:N", alt.Tooltip("value:Q", format=".2f")],
        )
    )
    st.altair_chart((hist + rules).properties(height=280), use_container_width=True)

    st.markdown("##### Bottom-up against top-down")
    gap = (result.p50 - topdown_value) / topdown_value if topdown_value else 0
    d1, d2, d3 = st.columns(3)
    d1.metric("Bottom-up (pipeline)", money(result.p50))
    d2.metric("Top-down (trend)", money(topdown_value))
    d3.metric("Divergence", f"{gap:+.1%}")
    st.caption(
        "The top-down model uses none of the pipeline — just seasonally "
        "decomposed history. When the two agree, that agreement is evidence. "
        "When they diverge, the gap is the thing to explain."
    )

    hist_df = actuals.assign(bookings=actuals.bookings / 1e6)
    bars = (
        alt.Chart(hist_df)
        .mark_bar(color=MUTED, opacity=0.75)
        .encode(
            x=alt.X("close_quarter:N", title=None),
            y=alt.Y("bookings:Q", title="Bookings (M)"),
            tooltip=["close_quarter", alt.Tooltip("bookings:Q", format=".2f")],
        )
    )
    fc = pd.DataFrame({
        "close_quarter": [current_q],
        "bookings": [result.p50 / 1e6],
        "low": [result.p10 / 1e6],
        "high": [result.p90 / 1e6],
    })
    fc_bar = alt.Chart(fc).mark_bar(color=ACCENT).encode(
        x="close_quarter:N", y="bookings:Q"
    )
    fc_rule = alt.Chart(fc).mark_rule(size=2, color="#1B2417").encode(
        x="close_quarter:N", y="low:Q", y2="high:Q"
    )
    st.altair_chart((bars + fc_bar + fc_rule).properties(height=260),
                    use_container_width=True)

# ---------------------------------------------------------------- Pipeline --
with tabs[1]:
    st.subheader("What the number is made of")

    closed = opps[(~opps.is_open) & opps.in_window]
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Open deals", f"{len(live):,}")
    p2.metric("Open pipeline", money(live.amount.sum()))
    p3.metric("Coverage vs quota", f"{live.amount.sum()/quota:.1f}×")
    p4.metric("Historical win rate", f"{closed.won.astype('boolean').mean():.0%}")

    by_seg = (
        live.assign(expected=live.amount * live.win_p)
        .groupby("segment", as_index=False)
        .agg(deals=("opp_id", "count"), pipeline=("amount", "sum"),
             expected=("expected", "sum"), win_p=("win_p", "mean"))
    )
    by_terr = (
        live.assign(expected=live.amount * live.win_p)
        .groupby("territory", as_index=False)
        .agg(deals=("opp_id", "count"), pipeline=("amount", "sum"),
             expected=("expected", "sum"), win_p=("win_p", "mean"))
        .sort_values("expected", ascending=False)
    )

    left, right = st.columns(2)
    with left:
        st.markdown("##### By segment")
        st.dataframe(
            by_seg.style.format({
                "pipeline": "{:,.0f}", "expected": "{:,.0f}", "win_p": "{:.1%}"
            }),
            use_container_width=True, hide_index=True,
        )
    with right:
        st.markdown("##### By territory")
        st.dataframe(
            by_terr.style.format({
                "pipeline": "{:,.0f}", "expected": "{:,.0f}", "win_p": "{:.1%}"
            }),
            use_container_width=True, hide_index=True,
        )

    st.markdown("##### How late deals actually close")
    slip = estimate_slip(opps, ANCHOR)
    slip_df = pd.DataFrame({"days": slip})
    st.altair_chart(
        alt.Chart(slip_df)
        .mark_bar(color=WARN, opacity=0.8)
        .encode(
            alt.X("days:Q", bin=alt.Bin(maxbins=50),
                  title="Days later than the rep predicted"),
            alt.Y("count()", title="Deals"),
        )
        .properties(height=220),
        use_container_width=True,
    )
    st.caption(
        f"Median slip is {np.median(slip):.0f} days and the 90th percentile is "
        f"{np.percentile(slip, 90):.0f}. The distribution is lopsided — deals "
        "slip late far more than they close early — so the simulation bootstraps "
        "from it rather than assuming a symmetric error."
    )

# ------------------------------------------------------------------- Model --
with tabs[2]:
    st.subheader("Can the probabilities be trusted?")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC AUC", f"{report.roc_auc:.3f}", f"oracle {report.oracle_roc_auc:.3f}")
    m2.metric("Signal recovered", f"{report.signal_recovered:.0%}")
    m3.metric("Brier score", f"{report.brier:.4f}", f"{report.brier_skill:+.1%} vs base rate")
    m4.metric("Test deals", f"{report.n_test:,}")

    st.caption(
        "Deals are decided by a weighted coin, so no model can reach AUC 1.0. "
        "The oracle — which knows each deal's eventual path — sets the real "
        "ceiling, and the model captures most of the distance to it."
    )

    st.markdown("##### Calibration")
    cal = report.calibration
    line = alt.Chart(cal).mark_line(point=True, color=ACCENT).encode(
        x=alt.X("predicted:Q", title="Predicted win probability",
                scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("observed:Q", title="Observed win rate",
                scale=alt.Scale(domain=[0, 1])),
        tooltip=["bin", alt.Tooltip("predicted:Q", format=".2f"),
                 alt.Tooltip("observed:Q", format=".2f"), "deals"],
    )
    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
        strokeDash=[4, 4], color=MUTED
    ).encode(x="x:Q", y="y:Q")
    st.altair_chart((diag + line).properties(height=300), use_container_width=True)
    st.caption(
        "Points on the diagonal mean the probabilities are honest: of the deals "
        "the model calls 40%, about 40% land. That property is what makes them "
        "safe to add up into a forecast."
    )

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("##### What drives a win")
        imp = report.importance.head(8)
        st.altair_chart(
            alt.Chart(imp).mark_bar(color=ACCENT).encode(
                x=alt.X("importance:Q", title="Permutation importance (AUC drop)"),
                y=alt.Y("feature:N", sort="-x", title=None),
            ).properties(height=260),
            use_container_width=True,
        )
    with c_right:
        st.markdown("##### Predicted vs observed by driver")
        driver = st.selectbox("Driver", ["source", "segment", "territory", "industry"])
        st.dataframe(
            group_effects(model, train, driver).style.format({
                "predicted": "{:.1%}", "observed": "{:.1%}", "gap": "{:+.1%}"
            }),
            use_container_width=True, hide_index=True,
        )

# ---------------------------------------------------------------- Backtest --
with tabs[3]:
    st.subheader("How wrong has this been before?")

    if bt.empty:
        st.warning("Not enough closed history to backtest.")
    else:
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Mean error", f"{bt_stats['mape']:.0%}")
        b2.metric("Median error", f"{bt_stats['median_ape']:.0%}")
        b3.metric("Interval coverage", f"{bt_stats['coverage']:.0%}", "target 80%")
        b4.metric("Bias", money(bt_stats["bias"]))

        st.caption(
            "Each quarter is re-forecast from scratch using only what was known "
            "on its first day — the model is refitted, the pipeline is rebuilt as "
            "it stood then. Coverage matters as much as error: if the P10–P90 "
            "band is honest, about 80% of quarters land inside it."
        )

        plot = bt.assign(
            actual=bt.actual / 1e6, p50=bt.p50 / 1e6,
            p10=bt.p10 / 1e6, p90=bt.p90 / 1e6,
        )
        band = alt.Chart(plot).mark_rule(size=3, color=MUTED, opacity=0.6).encode(
            x=alt.X("quarter:N", title=None), y="p10:Q", y2="p90:Q"
        )
        pred = alt.Chart(plot).mark_point(
            size=110, filled=True, color=ACCENT
        ).encode(x="quarter:N", y=alt.Y("p50:Q", title="Bookings (M)"),
                 tooltip=["quarter", alt.Tooltip("p50:Q", format=".2f")])
        act = alt.Chart(plot).mark_point(
            size=130, shape="diamond", filled=True, color=WARN
        ).encode(x="quarter:N", y="actual:Q",
                 tooltip=["quarter", alt.Tooltip("actual:Q", format=".2f")])
        st.altair_chart((band + pred + act).properties(height=300),
                        use_container_width=True)
        st.caption("Green = forecast P50, pink = what actually happened, bar = P10–P90.")

        show = bt.copy()
        for c in ["actual", "p10", "p50", "p90", "error"]:
            show[c] = show[c] / 1e6
        st.dataframe(
            show.style.format({
                "actual": "{:.2f}M", "p10": "{:.2f}M", "p50": "{:.2f}M",
                "p90": "{:.2f}M", "error": "{:+.2f}M", "ape": "{:.1%}",
            }),
            use_container_width=True, hide_index=True,
        )

# ---------------------------------------------------------------- Coverage --
with tabs[4]:
    st.subheader("Where should the hours go?")

    a1, a2, a3 = st.columns(3)
    a1.metric("Capacity", f"{alloc_summary['capacity_hours']:,.0f} h")
    a2.metric("Uplift from reallocating", money(alloc_summary["uplift"]),
              f"{alloc_summary['uplift_pct']:+.1%}")
    a3.metric("Undecided revenue", money(alloc_table.swing.sum()))

    st.caption(
        "Attention is worth most where the outcome is still in doubt. A deal at "
        "95% closes whether or not anyone works it; a deal at 5% will not close "
        "either way. The value genuinely in play is amount × p × (1−p), and that "
        "is what the allocation maximises under diminishing returns."
    )

    shift = alloc_table.assign(delta=alloc_table.hours_delta).nlargest(8, "uplift")
    st.altair_chart(
        alt.Chart(shift)
        .mark_bar()
        .encode(
            x=alt.X("delta:Q", title="Recommended change in hours"),
            y=alt.Y("territory:N", sort="-x", title=None),
            color=alt.Color("segment:N", legend=alt.Legend(orient="top", title=None)),
            tooltip=["territory", "segment", alt.Tooltip("delta:Q", format=".0f"),
                     alt.Tooltip("uplift:Q", format=",.0f")],
        )
        .properties(height=280),
        use_container_width=True,
    )

    st.dataframe(
        alloc_table[[
            "territory", "segment", "deals", "pipeline", "expected_value",
            "swing", "current_hours", "recommended_hours", "hours_delta", "uplift",
        ]].style.format({
            "pipeline": "{:,.0f}", "expected_value": "{:,.0f}", "swing": "{:,.0f}",
            "current_hours": "{:.0f}", "recommended_hours": "{:.0f}",
            "hours_delta": "{:+.0f}", "uplift": "{:,.0f}",
        }),
        use_container_width=True, hide_index=True,
    )

    st.markdown("##### The 20 deals that decide the quarter")
    share = triage.swing_share.sum()
    st.caption(
        f"These hold {share:.0%} of all undecided revenue in the pipeline. "
        "Reallocating hours across territories is worth little — the status quo "
        "is already close to optimal at that grain. The concentration is here."
    )
    st.dataframe(
        triage[[
            "opp_id", "territory", "segment", "source", "amount", "win_p",
            "swing", "swing_share", "expected_close_date",
        ]].style.format({
            "amount": "{:,.0f}", "win_p": "{:.1%}", "swing": "{:,.0f}",
            "swing_share": "{:.1%}",
        }),
        use_container_width=True, hide_index=True,
    )

# ------------------------------------------------------------------- Brief --
with tabs[5]:
    st.subheader("Executive brief")

    ctx = {
        "quarter": current_q,
        "p50": result.p50, "p10": result.p10, "p90": result.p90,
        "topdown": topdown_value,
        "committed": result.committed,
        "pipeline": float(live.amount.sum()),
        "n_deals": int(len(live)),
        "quota": quota,
        "p_quota": result.probability_of_quota(),
        "mape": bt_stats.get("mape", float("nan")),
        "coverage": bt_stats.get("coverage", float("nan")),
        "bt_quarters": bt_stats.get("quarters", 0),
        "top_cell": f"{alloc_table.iloc[0].territory} {alloc_table.iloc[0].segment}",
        "top_cell_swing": float(alloc_table.iloc[0].swing),
        "top_deals_share": float(triage.swing_share.sum()),
    }

    brief = generate_brief(ctx, use_claude=use_claude)
    if brief.error:
        st.info(brief.error)
    st.markdown(f"> {brief.text}")
    st.caption(
        f"Written by: **{brief.source}**. Every figure above is computed by the "
        "model and passed in already rounded — the writer is given facts to "
        "narrate, never a pipeline to do arithmetic on."
    )

    with st.expander("The figures the writer was given"):
        st.code("\n".join(f"{k}: {v}" for k, v in ctx.items()))
