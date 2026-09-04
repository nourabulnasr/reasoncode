"""EVAL GATE for ReasonCode.

CLAIM_UNDER_TEST: "The deterministic grounding gate (src/grounding_gate.py)
correctly distinguishes properly-grounded adverse-action narratives from
narratives that omit a required SHAP-selected reason or invent/hallucinate
a reason that was never selected -- i.e. it does the exact job the
'Accurate Ensembles, Fragile Narratives' paper (cited in validation.md)
found real LLM-based systems fail at without a verification step."

Why this is the right claim to test (and not "the LLM never hallucinates"):
this build could not get a real local causal LM to run in the sandbox it
was built in (see NOTES.md for the reproducible crash). The part of
ReasonCode's design that is actually novel and testable independent of
which generator produced the text is the gate itself -- so that's what
gets measured, on real applicants pulled from the real trained model and
real SHAP attributions, not synthetic feature vectors.

Method:
1. Train the real model, run real SHAP, pull top-3 reasons for 50 real
   declined test-set applicants.
2. For each applicant, build a GROUNDED narrative (template_narrative --
   guaranteed correct by construction) and two CORRUPTED variants:
   - OMISSION: drop one required reason from the narrative.
   - INVENTION: add a sentence mentioning a reason that was never selected
     for this applicant (simulating an LLM hallucinating an extra reason).
   This gives 150 labeled examples (50 grounded, 50 omission, 50 invention)
   with a known ground-truth label the gate did not see.
3. Score: does the gate correctly flag every corrupted example as failed,
   and correctly pass every grounded example?
"""

import json
import random
from datetime import datetime, timezone
from pathlib import Path

from src.data import load_gmsc
from src.explain import build_explainer, top_reasons_for_applicant
from src.generate import template_narrative
from src.grounding_gate import check_grounding
from src.model import declined_mask, train
from src.taxonomy import CANONICAL_REASONS

N_APPLICANTS = 50
SEED = 42
RESULTS_PATH = Path(__file__).resolve().parent.parent / "eval_results.json"


def corrupt_by_omission(reasons):
    """Drop the last required reason from what gets rendered, but the gate
    is still told all of `reasons` were required -- simulates an LLM that
    forgot to mention a real reason."""
    kept = reasons[:-1] if len(reasons) > 1 else []
    return template_narrative(kept)


def corrupt_by_invention(reasons, rng):
    """Render the correct narrative, then splice in a sentence about a
    reason that was NOT selected for this applicant -- simulates an LLM
    hallucinating an extra reason."""
    correct = template_narrative(reasons)
    unused = [r for r in CANONICAL_REASONS if r not in reasons]
    invented = rng.choice(unused)
    return correct + f" We also note: {invented.reg_b_text}."


def run_eval():
    print("Training model and building SHAP explainer...")
    tm = train(seed=SEED)
    explainer = build_explainer(tm.model)
    declined = declined_mask(tm.model, tm.X_test)
    sample_idx = list(tm.X_test[declined].index[:N_APPLICANTS])
    assert len(sample_idx) == N_APPLICANTS, f"expected {N_APPLICANTS} declined applicants, got {len(sample_idx)}"

    rng = random.Random(SEED)
    examples = []  # (label, gate_result.passed)
    for idx in sample_idx:
        row = tm.X_test.loc[idx]
        ar = top_reasons_for_applicant(explainer, row, k=3)
        if len(ar.reasons) < 2:
            # need at least 2 reasons to construct a meaningful omission case
            continue

        grounded_text = template_narrative(ar.reasons)
        omission_text = corrupt_by_omission(ar.reasons)
        invention_text = corrupt_by_invention(ar.reasons, rng)

        examples.append(("grounded", check_grounding(grounded_text, ar.reasons).passed))
        examples.append(("omission", check_grounding(omission_text, ar.reasons).passed))
        examples.append(("invention", check_grounding(invention_text, ar.reasons).passed))

    n = len(examples)
    grounded = [passed for label, passed in examples if label == "grounded"]
    omission = [passed for label, passed in examples if label == "omission"]
    invention = [passed for label, passed in examples if label == "invention"]

    # Correct behavior: grounded examples should PASS (True), corrupted
    # examples should FAIL (False, i.e. gate correctly caught them).
    grounded_correct = sum(grounded)  # want all True
    omission_caught = sum(1 for p in omission if not p)  # want all False (caught)
    invention_caught = sum(1 for p in invention if not p)  # want all False (caught)

    total_correct = grounded_correct + omission_caught + invention_caught
    accuracy = total_correct / n if n else 0.0

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "claim_under_test": (
            "The deterministic grounding gate correctly passes grounded "
            "narratives and correctly rejects narratives with an omitted "
            "or invented/hallucinated reason."
        ),
        "n_applicants": len(sample_idx),
        "n_examples": n,
        "grounded_pass_rate": grounded_correct / len(grounded) if grounded else None,
        "omission_catch_rate": omission_caught / len(omission) if omission else None,
        "invention_catch_rate": invention_caught / len(invention) if invention else None,
        "overall_accuracy": accuracy,
    }

    print(json.dumps(result, indent=2))

    # Append-only: never overwrite prior runs.
    prior = []
    if RESULTS_PATH.exists():
        prior = json.loads(RESULTS_PATH.read_text())
        if not isinstance(prior, list):
            prior = [prior]
    prior.append(result)
    RESULTS_PATH.write_text(json.dumps(prior, indent=2))
    print(f"\nAppended result to {RESULTS_PATH}")
    return result


if __name__ == "__main__":
    run_eval()
