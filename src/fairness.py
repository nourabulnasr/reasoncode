"""Batch fairness audit: four-fifths (80%) rule on decline rates by age group.

ECOA/Regulation B treats age as a protected basis. This module runs the
classic disparate-impact screen (EEOC four-fifths rule, also standard
practice in credit fair-lending audits) comparing the decline rate for an
older-applicant group against a younger reference group. It runs on the
FULL test set (cheap -- no LLM involved), independent of how many
narratives were sampled for the generation step.
"""

from dataclasses import dataclass

import pandas as pd
from fairlearn.metrics import MetricFrame, selection_rate


@dataclass
class FairnessReport:
    group_decline_rates: dict  # group label -> decline rate
    four_fifths_ratio: float  # min(group selection rate) / max(group selection rate)
    passes_four_fifths_rule: bool  # ratio >= 0.8


def four_fifths_audit(y_pred, age: pd.Series, threshold_age: int = 40) -> FairnessReport:
    """Split applicants into two age groups at `threshold_age` and compare
    decline (predicted-positive) rates. Ratio of the lower to the higher
    rate below 0.8 signals adverse impact under the standard four-fifths
    rule.
    """
    group = pd.Series(
        ["age_under_%d" % threshold_age if a < threshold_age else "age_%d_plus" % threshold_age for a in age],
        index=age.index,
    )
    mf = MetricFrame(metrics=selection_rate, y_true=y_pred, y_pred=y_pred, sensitive_features=group)
    rates = mf.by_group.to_dict()
    ratio = min(rates.values()) / max(rates.values()) if max(rates.values()) > 0 else 0.0
    return FairnessReport(
        group_decline_rates=rates,
        four_fifths_ratio=float(ratio),
        passes_four_fifths_rule=ratio >= 0.8,
    )
