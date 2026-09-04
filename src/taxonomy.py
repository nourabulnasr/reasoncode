"""
Fixed regulatory reason-code taxonomy for ECOA/Regulation B adverse action
notices.

Regulation B (12 CFR 1002, Appendix C) requires a creditor to disclose the
"principal reason(s)" a credit application was denied, using specific and
non-generic language. Appendix C provides a sample list of reasons a
creditor may check off; the entries below are drawn from that sample list
(condensed to the ones that map onto Give Me Some Credit's feature set) plus
a short canonical keyword phrase a generator can weave into a sentence.

Two GMSC columns are deliberately EXCLUDED from ever appearing as a reason:
- age                 -> protected basis under ECOA (12 CFR 1002.6(b)(2)).
                         Used ONLY for the fairness audit, never surfaced
                         to an applicant as "why you were declined."
- NumberOfDependents   -> proxy for marital/familial status, a basis ECOA
                         also restricts creditors from adversely weighing.
                         Excluded from reason codes for the same reason.

Design note on keyword uniqueness (this mattered in practice -- see
NOTES.md): the grounding gate (src/grounding_gate.py) flags a reason as
"invented" if ANY of its keywords appear in a narrative that wasn't
supposed to mention it, and flags a reason as "missing" if NONE of its
keywords appear in a narrative that was supposed to mention it. Two
GMSC columns (the 30-59 and 60-89 day late buckets) map to the
*identical* Regulation B principal-reason text -- Reg B doesn't
distinguish the bucket -- so they are merged into one canonical
ReasonCode here (both dict keys point at the same object).

Each `keywords` entry is deliberately chosen as a literal substring of
its own `reg_b_text` (not an independently-styled paraphrase -- an
earlier version of this file used paraphrased keywords and it silently
broke the coverage check for every reason that wasn't first in a
narrative's list, since only the lead reason's paraphrase actually got
written into the generated text; see NOTES.md for the concrete repro).
Making each keyword a substring of its own reg_b_text means: whenever a
generator prints a reason's reg_b_text verbatim (which the template
generator always does, for every reason, not just the first), coverage
is satisfied by construction. Each keyword is also checked (in
test_taxonomy.py) to NOT be a substring of, or contain as a substring,
any other reason's keyword -- so two genuinely different reasons can
never falsely trigger each other in the invention check.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ReasonCode:
    reg_b_text: str
    # Distinctive phrase(s) unique across the whole taxonomy -- both what
    # the template generator writes and what the grounding gate looks for.
    keywords: tuple


# GMSC columns that are legally off-limits as adverse-action reasons.
PROTECTED_FEATURES = frozenset({"age", "NumberOfDependents"})

_REVOLVING_UTILIZATION = ReasonCode(
    reg_b_text="Excessive use of available revolving credit lines",
    keywords=("revolving credit lines",),
)
_RECENT_DELINQUENCY = ReasonCode(
    reg_b_text="Delinquent payment history in the past 12 months (30-89 days past due)",
    keywords=("30-89 days past due",),
)
_DEBT_RATIO = ReasonCode(
    reg_b_text="Excessive obligations relative to income",
    keywords=("obligations relative to income",),
)
_MONTHLY_INCOME = ReasonCode(
    reg_b_text="Income insufficient for amount of credit requested",
    keywords=("income insufficient for amount of credit",),
)
_OPEN_CREDIT_LINES = ReasonCode(
    reg_b_text="Insufficient number of credit references",
    keywords=("insufficient number of credit references",),
)
_SERIOUS_DELINQUENCY_90 = ReasonCode(
    reg_b_text="Serious delinquency (90+ days past due) on past or present credit obligations",
    keywords=("90+ days past due",),
)
_REAL_ESTATE_LOANS = ReasonCode(
    reg_b_text="Excessive number of real-estate-secured obligations",
    keywords=("real-estate-secured obligations",),
)

# Feature -> canonical ReasonCode. Two features intentionally share one
# object (see module docstring).
TAXONOMY = {
    "RevolvingUtilizationOfUnsecuredLines": _REVOLVING_UTILIZATION,
    "NumberOfTime30-59DaysPastDueNotWorse": _RECENT_DELINQUENCY,
    "NumberOfTime60-89DaysPastDueNotWorse": _RECENT_DELINQUENCY,
    "DebtRatio": _DEBT_RATIO,
    "MonthlyIncome": _MONTHLY_INCOME,
    "NumberOfOpenCreditLinesAndLoans": _OPEN_CREDIT_LINES,
    "NumberOfTimes90DaysLate": _SERIOUS_DELINQUENCY_90,
    "NumberRealEstateLoansOrLines": _REAL_ESTATE_LOANS,
}

# Every distinct canonical reason (de-duplicated by identity), for the
# grounding gate's "did you invent something extra" scan.
CANONICAL_REASONS = list({id(rc): rc for rc in TAXONOMY.values()}.values())


def reason_for(feature: str) -> ReasonCode | None:
    """Look up the fixed taxonomy entry for a GMSC feature, or None if the
    feature is protected/unmapped and must never be surfaced as a reason."""
    if feature in PROTECTED_FEATURES:
        return None
    return TAXONOMY.get(feature)
