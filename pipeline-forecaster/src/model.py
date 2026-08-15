"""
Win-probability model.

A forecast built from uncalibrated scores is worth very little. If the model
says 70% on a hundred deals, roughly seventy of them need to land, or every
downstream number — the quarter forecast, the confidence band, the allocation
of selling hours — inherits the bias. So the model is judged first on
calibration (Brier score, reliability curve) and only then on ranking (AUC).

The classifier is a gradient-boosted tree wrapped in isotonic calibration,
fitted on as-of snapshots and validated on a *later* slice of time than it was
trained on, because a random split would let it learn from the future.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, check_no_leakage

TEST_FRACTION = 0.25  # most recent quarter-ish of closed deals, held out by date


@dataclass
class ModelReport:
    """Everything needed to judge whether the model can be trusted."""

    roc_auc: float
    pr_auc: float
    brier: float
    log_loss: float
    base_rate: float
    brier_skill: float  # vs always predicting the base rate
    oracle_roc_auc: float  # ceiling: an oracle that knows each deal's full path
    n_train: int
    n_test: int
    calibration: pd.DataFrame
    importance: pd.DataFrame

    def summary(self) -> str:
        return (
            f"ROC AUC {self.roc_auc:.3f} (oracle {self.oracle_roc_auc:.3f}) | "
            f"PR AUC {self.pr_auc:.3f} | "
            f"Brier {self.brier:.4f} (skill {self.brier_skill:+.1%} vs base rate) | "
            f"n_train {self.n_train:,} n_test {self.n_test:,}"
        )

    @property
    def signal_recovered(self) -> float:
        """
        Share of the achievable ranking signal the model actually captures.

        Deals are decided by a coin weighted by their win probability, so no
        model can reach AUC 1.0. Measuring against the oracle instead of against
        1.0 is the difference between "0.69 is mediocre" and "0.69 is most of
        what was there to get".
        """
        lift = self.oracle_roc_auc - 0.5
        return (self.roc_auc - 0.5) / lift if lift > 0 else 0.0


def _build_pipeline(kind: str = "gbm", fast: bool = False) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ]
    )
    if kind == "logistic":
        clf = LogisticRegression(max_iter=2000, C=1.0)
    else:
        clf = HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.06,
            max_iter=90 if fast else 280,
            min_samples_leaf=40,
            l2_regularization=1.0,
            random_state=7,
        )
    return Pipeline([("pre", pre), ("clf", clf)])


def calibration_table(y_true: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Reliability curve: predicted probability vs observed frequency."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin": f"{edges[b]:.1f}–{edges[b + 1]:.1f}",
                "predicted": float(p[mask].mean()),
                "observed": float(y_true[mask].mean()),
                "deals": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def temporal_split(frame: pd.DataFrame, test_fraction: float = TEST_FRACTION):
    """
    Split by close date, not at random.

    A random split lets the model train on deals that closed after the ones it
    is tested on, which quietly inflates every metric.
    """
    order = frame.sort_values("_close_date")
    cut = int(len(order) * (1 - test_fraction))
    return order.iloc[:cut], order.iloc[cut:]


def train_win_model(
    frame: pd.DataFrame, kind: str = "gbm", fast: bool = False
) -> tuple[Pipeline, ModelReport]:
    """
    Fit, calibrate and score the win-probability model.

    `fast` trims the tree count and calibration folds. It exists for the
    backtest, which refits the model once per historical quarter and would
    otherwise take long enough that nobody runs it.
    """
    check_no_leakage(frame[[c for c in frame.columns if not c.startswith("_")]])

    train, test = temporal_split(frame)
    X_train, y_train = train[FEATURES], train.won.to_numpy()
    X_test, y_test = test[FEATURES], test.won.to_numpy()

    base = _build_pipeline(kind, fast=fast)
    # Isotonic calibration on internal CV folds — the reason the probabilities
    # can be summed into a forecast at all.
    model = CalibratedClassifierCV(base, method="isotonic", cv=3 if fast else 5)
    model.fit(X_train, y_train)

    p = model.predict_proba(X_test)[:, 1]
    base_rate = float(y_train.mean())
    brier = float(brier_score_loss(y_test, p))
    brier_base = float(brier_score_loss(y_test, np.full_like(p, base_rate)))

    imp = permutation_importance(
        model, X_test, y_test, n_repeats=2 if fast else 8, random_state=7,
        scoring="roc_auc",
    )
    importance = (
        pd.DataFrame({"feature": FEATURES, "importance": imp.importances_mean})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    # An oracle scoring deals by the generator's own probability — which knows
    # the eventual stage and latent quality — sets the ceiling for this data.
    oracle = (
        float(roc_auc_score(y_test, test._true_win_p.to_numpy()))
        if "_true_win_p" in test.columns
        else float("nan")
    )

    report = ModelReport(
        roc_auc=float(roc_auc_score(y_test, p)),
        pr_auc=float(average_precision_score(y_test, p)),
        brier=brier,
        log_loss=float(log_loss(y_test, p)),
        base_rate=base_rate,
        brier_skill=float(1 - brier / brier_base) if brier_base > 0 else 0.0,
        oracle_roc_auc=oracle,
        n_train=len(train),
        n_test=len(test),
        calibration=calibration_table(y_test, p),
        importance=importance,
    )
    return model, report


def score(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    """Calibrated win probability for each row."""
    if frame.empty:
        return np.array([])
    return model.predict_proba(frame[FEATURES])[:, 1]


def group_effects(model: Pipeline, frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Predicted vs observed win rate for each level of a driver.

    The obvious alternative — re-score every deal as if it came from each level
    in turn — reads as a cleaner causal story but is not trustworthy here.
    Segment determines deal size and cycle length, so relabelling a small fast
    deal as Enterprise produces a row that could not exist, and the model's
    answer to an impossible question tells you nothing. Comparing predictions
    against outcomes within each real group stays on the data manifold and
    answers the question that matters anyway: does the model reproduce the gaps
    the business already knows are there?
    """
    work = frame.copy()
    work["predicted"] = score(model, work)
    out = (
        work.groupby(column)
        .agg(
            predicted=("predicted", "mean"),
            observed=("won", "mean"),
            deals=("opp_id", "count"),
        )
        .reset_index()
        .sort_values("observed", ascending=False)
        .reset_index(drop=True)
    )
    out["gap"] = out.predicted - out.observed
    return out
