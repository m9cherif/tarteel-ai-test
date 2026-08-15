"""
The executive summary.

A forecast that a VP of Sales has to decode from percentiles does not get read.
This module turns the model's output into the paragraph someone would actually
open the deck with — the number, the confidence, and the one thing to do about
it.

Two rules govern what happens here, and they are the reason this is a small
module rather than a clever one:

1. **The model never invents a figure.** Every number in the brief is computed
   in `forecast.py` and `allocation.py` and passed in already rounded. Claude is
   asked to write prose around a supplied set of facts, not to do arithmetic on
   a pipeline it cannot see. Anything it cannot ground in the facts given, it
   omits.

2. **It is optional.** With no API key — or no `anthropic` package installed —
   the same brief is assembled from a template. The app must not have a hard
   dependency on a paid API to show a forecast, so the AI layer is an upgrade to
   the writing, never a prerequisite for the analysis.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

MODEL = "claude-opus-5"

SYSTEM_PROMPT = """\
You are a revenue operations analyst writing the opening paragraphs of a \
quarterly business review for a VP of Sales.

You will be given a set of figures from a pipeline forecasting model. Write a \
brief of 120-180 words covering, in this order:

1. Where the quarter is likely to land, and how confident the model is.
2. The single most important risk or opportunity in the numbers.
3. One concrete recommendation the VP can act on this week.

Rules:
- Use only the figures provided. Do not introduce numbers that are not in the \
input, and do not recompute or adjust the ones that are.
- If the bottom-up and top-down forecasts disagree, say so plainly and say \
which one you would trust here and why.
- Write in plain sentences. No bullet points, no headers, no preamble such as \
"Here is the brief". Start with the finding.
- Be direct about bad news. A forecast brief that buries a miss is worthless.\
"""


@dataclass
class Brief:
    """The written summary plus where it came from."""

    text: str
    source: str  # "claude" or "template"
    error: str | None = None


def _facts(ctx: dict) -> str:
    """Flatten the computed figures into the fact sheet the model is given."""
    lines = [
        f"Quarter: {ctx['quarter']}",
        f"Bottom-up forecast (P50): {ctx['p50']/1e6:.2f}M",
        f"80% confidence range: {ctx['p10']/1e6:.2f}M to {ctx['p90']/1e6:.2f}M",
        f"Top-down forecast from historical trend: {ctx['topdown']/1e6:.2f}M",
        f"Already closed-won this quarter: {ctx['committed']/1e6:.2f}M",
        f"Open pipeline value: {ctx['pipeline']/1e6:.2f}M",
        f"Open deals: {ctx['n_deals']}",
        f"Quota: {ctx['quota']/1e6:.2f}M",
        f"Probability of hitting quota: {ctx['p_quota']:.0%}",
        f"Model backtest: {ctx['mape']:.0%} mean error across "
        f"{ctx['bt_quarters']} past quarters, "
        f"{ctx['coverage']:.0%} of them inside the 80% band",
    ]
    if ctx.get("top_cell"):
        lines.append(
            f"Territory/segment with the most undecided revenue: {ctx['top_cell']} "
            f"({ctx['top_cell_swing']/1e6:.2f}M at stake)"
        )
    if ctx.get("top_deals_share"):
        lines.append(
            f"Concentration: the top 20 deals hold {ctx['top_deals_share']:.0%} "
            "of all undecided revenue"
        )
    return "\n".join(lines)


def _template_brief(ctx: dict) -> str:
    """
    The fallback brief. Deterministic, unglamorous, and always available.

    Deliberately written to be genuinely usable rather than a stub, because on
    a machine with no API key this is the only brief anyone sees.
    """
    gap = ctx["p50"] - ctx["quota"]
    verdict = "above" if gap >= 0 else "below"
    spread = ctx["p90"] - ctx["p10"]

    divergence = (ctx["p50"] - ctx["topdown"]) / ctx["topdown"] if ctx["topdown"] else 0
    if abs(divergence) < 0.05:
        agreement = (
            "The top-down trend model agrees within 5%, which is the strongest "
            "evidence available that the number is real."
        )
    else:
        direction = "above" if divergence > 0 else "below"
        agreement = (
            f"The bottom-up number sits {abs(divergence):.0%} {direction} the "
            f"top-down trend forecast of {ctx['topdown']/1e6:.2f}M. That gap is "
            "worth explaining before the number is committed."
        )

    parts = [
        f"{ctx['quarter']} is tracking to {ctx['p50']/1e6:.2f}M, "
        f"{abs(gap)/1e6:.2f}M {verdict} the {ctx['quota']/1e6:.2f}M quota, with a "
        f"{ctx['p_quota']:.0%} chance of clearing it. The 80% range runs "
        f"{ctx['p10']/1e6:.2f}M to {ctx['p90']/1e6:.2f}M — a "
        f"{spread/1e6:.2f}M spread, which is the honest width of the uncertainty "
        "this early.",
        agreement,
        f"{ctx['committed']/1e6:.2f}M is already closed and no longer at risk; the "
        f"rest depends on {ctx['n_deals']} open deals worth "
        f"{ctx['pipeline']/1e6:.2f}M.",
    ]

    if ctx.get("top_deals_share"):
        parts.append(
            f"The quarter is more concentrated than it looks: the top 20 deals "
            f"carry {ctx['top_deals_share']:.0%} of all undecided revenue. Working "
            "that list is worth more this week than broad pipeline coverage."
        )

    parts.append(
        f"Treat the range, not the midpoint, as the forecast: across "
        f"{ctx['bt_quarters']} backtested quarters this model missed by "
        f"{ctx['mape']:.0%} on average."
    )
    return " ".join(parts)


def generate_brief(ctx: dict, use_claude: bool = True) -> Brief:
    """
    Write the executive brief, preferring Claude and falling back to the template.

    Never raises: a failure to reach the API downgrades the writing, it does not
    take down the forecast.
    """
    if not use_claude:
        return Brief(_template_brief(ctx), "template")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        return Brief(
            _template_brief(ctx),
            "template",
            error="No ANTHROPIC_API_KEY set — using the built-in writer.",
        )

    try:
        import anthropic
    except ImportError:
        return Brief(
            _template_brief(ctx),
            "template",
            error="The anthropic package is not installed — using the built-in writer.",
        )

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=MODEL,
            # Thinking is on by default and shares this budget with the reply,
            # so the cap is set well above what 180 words needs.
            max_tokens=8000,
            # A short brief over supplied figures is a scoped task; low effort
            # keeps it fast and cheap without costing quality here.
            output_config={"effort": "low"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _facts(ctx)}],
        )

        # Check why generation stopped before touching content — on a refusal
        # the content list is empty or partial.
        if response.stop_reason == "refusal":
            return Brief(
                _template_brief(ctx),
                "template",
                error="The request was declined — using the built-in writer.",
            )

        text = "\n\n".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        if not text:
            return Brief(
                _template_brief(ctx),
                "template",
                error="Empty response — using the built-in writer.",
            )
        return Brief(text, "claude")

    except anthropic.RateLimitError:
        return Brief(
            _template_brief(ctx), "template", error="Rate limited — using the built-in writer."
        )
    except anthropic.APIError as exc:
        return Brief(
            _template_brief(ctx), "template", error=f"API error: {exc} — using the built-in writer."
        )
    except Exception as exc:  # network, auth, anything else
        return Brief(
            _template_brief(ctx), "template", error=f"{exc} — using the built-in writer."
        )
