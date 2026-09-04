# ReasonCode

A grounded adverse-action-notice generator for credit decisions.

When a lender's ML model declines a credit application, ECOA / Regulation
B requires the lender to tell the applicant the *specific, accurate*
principal reasons -- not a vague "your score was too low." Real fintechs
have received CFPB consent orders for getting this wrong. ReasonCode is a
small end-to-end pipeline that shows what a compliant version of that
step actually looks like, and -- the actually novel part -- mechanically
verifies that the generated letter didn't invent or omit a reason before
it ships.

**EVAL GATE VERDICT: CONFIRMED.** The grounding gate correctly caught
100% (50/50) of narratives with an omitted reason and 100% (50/50) with
an invented/hallucinated reason, while passing 100% (50/50) of correctly-
grounded narratives, tested on real declined applicants from the real
trained model. See `EVAL-FINDINGS.md` for the full method and
`eval_results.json` for the raw run.

## What it actually does

1. **Trains a real classifier** (LightGBM) on the public [Give Me Some
   Credit](https://www.kaggle.com/c/GiveMeSomeCredit) dataset (~150k
   labeled applicants) to predict serious delinquency risk. A predicted
   "high risk" applicant stands in for a declined credit application.
2. **Explains each decline individually** with real per-applicant SHAP
   (TreeExplainer) attribution -- not a global feature-importance chart.
3. **Maps SHAP-selected features to a fixed regulatory taxonomy**
   (`src/taxonomy.py`) modeled on Regulation B Appendix C's sample reason
   list. Two features that ECOA treats as protected/restricted bases
   (`age`, `NumberOfDependents`) are hard-excluded from ever appearing as
   a reason -- they're used only in the fairness audit below.
4. **Generates a narrative letter** from the selected reasons.
5. **Runs a deterministic grounding gate** (`src/grounding_gate.py`)
   against that narrative: does it mention every required reason, and
   nothing else? This is the mechanism the arXiv paper cited in
   `validation.md` ("Accurate Ensembles, Fragile Narratives") found real
   LLM-based systems need and don't have -- verification *after*
   generation, done mechanically, not by asking another LLM to judge
   itself.
6. **Runs a batch fairness audit** (four-fifths rule, via `fairlearn`) on
   decline rates by age group, across the *full* test set.

## A real scope cut, stated plainly

The original design called for a small local instruction-tuned LLM
(Qwen2.5-1.5B-class) to write the narrative. That code path
(`src/generate.py::llm_narrative`) is fully implemented, but this build
could not get any causal LM -- down to 135M parameters -- to load and run
reliably in the sandbox it was built in (repeatable Rust allocator
aborts / Windows "paging file too small" errors / segfaults across four
different loading strategies). Full repro details in `NOTES.md`. The
shipped demo instead uses `template_narrative()`, a deterministic
generator built directly from the same SHAP-selected reasons -- which,
by construction, always passes the grounding gate. The gate itself (the
actually novel, testable mechanism) is exercised for real either way; see
`EVAL-FINDINGS.md`.

## Quickstart (30 seconds)

```bash
pip install -r requirements.txt
python pipeline.py demo
```

This trains the model, scores the test set, picks 20 real declined
applicants, generates and grounding-checks a letter for each, prints one
sample letter, and prints the fairness audit -- all against the real
150k-row dataset already included in `data/cs-training.csv`. No API keys,
no downloads needed (`.env.example` documents this explicitly).

Run the test suite (real model, real SHAP, no mocks):

```bash
python -m pytest -q
```

Run the eval gate yourself:

```bash
python -m eval.eval_grounding_gate
```

## What a real fairness finding looks like

The demo's fairness audit is not a stub -- on this dataset it actually
fails the four-fifths rule (ratio ~0.40: applicants under 40 are declined
at roughly 2.5x the rate of applicants 40+). That's a genuine, expected
finding for a naive risk model trained without any fairness constraint --
younger applicants tend to have thinner credit files, which the model
reads as risk. ReasonCode doesn't fix this; it *surfaces* it, which is
the point of a fairness audit step in a real compliance pipeline.

## Repo layout

```
data/cs-training.csv       real GMSC dataset (~7.3MB, committed)
src/data.py                load + clean GMSC
src/model.py                LightGBM classifier
src/explain.py              per-applicant SHAP -> taxonomy reasons
src/taxonomy.py             fixed Reg-B-style reason taxonomy
src/generate.py              template + (untested) LLM narrative generators
src/grounding_gate.py       deterministic post-generation verifier
src/fairness.py             four-fifths rule audit
pipeline.py                  CLI: `python pipeline.py demo`
tests/                       pytest suite (real model, no mocks)
eval/eval_grounding_gate.py  the EVAL GATE script
eval_results.json            append-only raw eval history
EVAL-FINDINGS.md             eval method + verdict
NOTES.md                     the local-LLM sandbox investigation, in full
```

## What's missing / what a human needs to do to ship this

- Wire up a real small instruction LLM for `llm_narrative()` on a machine
  without this sandbox's memory ceiling, and re-run
  `eval/eval_grounding_gate.py`-style corruption testing against *real*
  LLM output (not just synthetic corruptions of the template output) to
  see how often an actual small model hallucinates before the gate catches
  it -- that's the more interesting number this build didn't get to.
- The taxonomy (`src/taxonomy.py`) is a reasonable approximation of
  Regulation B Appendix C language, not legal advice -- a real deployment
  needs compliance sign-off on the exact reason-code wording.
- GMSC's `SeriousDlqin2yrs` label is a proxy for "should decline," not a
  real underwriting decision -- swapping in a real lender's label/feature
  set is a bigger lift than this MVP.
- No persistence/API layer -- this is a pipeline script, not a service.
