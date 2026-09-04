"""Deterministic grounding gate for generated adverse-action narratives.

This is the mechanism the validation doc (and the arXiv "Accurate Ensembles,
Fragile Narratives" paper it cites) identified as the missing piece in prior
work: grounding must be VERIFIED after generation, not assumed, and that
verification must itself be mechanical/deterministic -- not another LLM call
judging faithfulness, which just moves the trust problem rather than solving
it.

The gate checks a generated narrative against the exact set of ReasonCode
objects SHAP selected for that applicant (src/explain.py), using only
substring/keyword matching over the full taxonomy:

1. COVERAGE: every allowed reason must actually be mentioned (at least one
   of its keywords present) -- catches omission of a real, required reason.
2. NO-INVENTION: no keyword belonging to a taxonomy reason that is NOT in
   the allowed set may appear -- catches the LLM inventing/hallucinating a
   reason that SHAP never actually selected for this applicant.

Both checks run on lower-cased text with simple substring search. This is
intentionally "dumb" -- that is the point: a mechanically verifiable gate
cannot itself hallucinate.
"""

from dataclasses import dataclass

from src.taxonomy import CANONICAL_REASONS, ReasonCode


@dataclass
class GateResult:
    passed: bool
    missing_reasons: list[str]  # reg_b_text of required reasons not found
    invented_reasons: list[str]  # reg_b_text of reasons mentioned but not allowed


def check_grounding(narrative: str, allowed_reasons: list[ReasonCode]) -> GateResult:
    text = narrative.lower()
    allowed_ids = {id(r) for r in allowed_reasons}

    missing = []
    for r in allowed_reasons:
        if not any(kw in text for kw in r.keywords):
            missing.append(r.reg_b_text)

    invented = []
    for r in CANONICAL_REASONS:
        if id(r) in allowed_ids:
            continue
        if any(kw in text for kw in r.keywords):
            invented.append(r.reg_b_text)

    passed = not missing and not invented
    return GateResult(passed=passed, missing_reasons=missing, invented_reasons=invented)
