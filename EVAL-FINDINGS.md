# EVAL-FINDINGS

**CLAIM_UNDER_TEST** (stated verbatim in `eval/eval_grounding_gate.py`):

> The deterministic grounding gate (`src/grounding_gate.py`) correctly
> distinguishes properly-grounded adverse-action narratives from
> narratives that omit a required SHAP-selected reason or invent/
> hallucinate a reason that was never selected -- i.e. it does the exact
> job the "Accurate Ensembles, Fragile Narratives" paper (cited in
> validation.md) found real LLM-based systems fail at without a
> verification step.

## Why this claim, not "the LLM never hallucinates"

This build could not get a real local causal LM to load and run reliably
in the sandbox it was built in (four independent loading strategies tried
across two models, all failed or were non-reproducible -- full repro
details in `NOTES.md`). The part of ReasonCode's design that is testable
independent of which generator produced the raw text is the gate itself,
so that is what this eval measures -- on real applicants from the real
trained model and real SHAP attributions, not synthetic feature vectors.

## Method

1. Train the real LightGBM model on the real GMSC test split, run real
   SHAP (`TreeExplainer`), pull the top-3 taxonomy reasons for 50 real
   declined test-set applicants.
2. For each applicant, build 3 labeled examples:
   - **grounded**: `template_narrative()` output (correct by construction).
   - **omission**: same narrative with the last required reason dropped
     from the text, while the gate is still told all reasons were
     required (simulates an LLM that forgot a real reason).
   - **invention**: the grounded narrative plus one extra sentence about a
     taxonomy reason that was NOT selected for this applicant (simulates
     an LLM hallucinating an extra reason).
3. Run `check_grounding()` on all 150 examples and compare its pass/fail
   verdict against the known ground-truth label.

## Result (run 1, only run -- see `eval_results.json` for the raw record)

| metric | value |
|---|---|
| grounded narratives correctly passed | 50/50 (100%) |
| omission narratives correctly caught | 50/50 (100%) |
| invention narratives correctly caught | 50/50 (100%) |
| overall accuracy | 150/150 (100%) |

No diagnose-and-fix attempts were needed -- the first run already met the
claim, so the two-attempt fix budget was not used.

## Honest caveat on why this result is expected, not surprising

`check_grounding()` is a deterministic substring-match function, not a
statistical classifier -- a 100% result on a well-formed eval set is the
*correct* outcome of a correctly-implemented mechanical check, not
evidence of some emergent robustness. The real engineering risk this eval
actually screens for is the taxonomy-design bug found and fixed during
the first live demo run (see git history / `src/taxonomy.py` docstring):
overlapping or non-substring keyword phrases silently broke coverage for
17/20 real narratives before the fix. This eval would have caught that
exact bug (the "grounded" row would have failed to hit 100%) -- and does
guard against it regressing, via `tests/test_taxonomy.py`'s keyword-
uniqueness assertions plus this eval's `grounded_pass_rate`.

## Verdict: **CONFIRMED**

The claim holds on the test constructed. The grounding gate correctly
passes all grounded narratives and correctly rejects all omission and
invention corruptions, on real applicants scored by the real trained
model.
