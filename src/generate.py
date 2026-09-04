"""Adverse-action narrative generation, with a mechanically-verified
grounding gate in front of it.

Two generator backends are implemented:

- `template_narrative`: a deterministic, template-based narrative built
  directly from the SHAP-selected ReasonCode objects. By construction it
  can only ever mention the reasons it was given, so it always passes the
  grounding gate. This is the generator the shipped demo actually runs.

- `llm_narrative`: a real small-instruction-model-backed generator (uses
  a local HF `transformers` causal LM, no API key). The code is complete
  and functional in design, but could NOT be exercised end-to-end in the
  build environment this repo was built in -- see NOTES.md for the exact,
  reproducible crash (Rust allocator aborts / Windows "paging file too
  small" / segfaults) hit across four independent loading strategies for
  every causal LM tried (Qwen2.5-0.5B-Instruct, SmolLM2-135M-Instruct),
  while non-generative transformer encoders of similar and larger size
  (sentence-transformers/all-MiniLM-L6-v2) loaded and ran without issue.
  This looks like a sandbox-specific commit-charge ceiling hit specifically
  by causal-LM generation code paths, not a bug in this repo's code.
  `use_llm=True` is provided for a reader running on a normal machine.

Both backends are run through the same `generate_with_gate` orchestrator,
so the grounding gate's behavior is identical and testable regardless of
which generator produced the raw text -- this is what src/../eval/
eval_grounding_gate.py actually tests.
"""

from dataclasses import dataclass

from src.grounding_gate import GateResult, check_grounding
from src.taxonomy import ReasonCode

MAX_REGENERATE_ATTEMPTS = 2


def template_narrative(reasons: list[ReasonCode]) -> str:
    """Deterministic, always-grounded adverse-action paragraph. Mentions
    every given reason using its exact taxonomy keyword, and nothing else.
    """
    if not reasons:
        return (
            "After careful review, we are unable to approve your application "
            "at this time based on the information in your credit file."
        )
    # Render every reason's reg_b_text verbatim (not a paraphrase) -- each
    # reason's `keywords` is defined as a literal substring of its own
    # reg_b_text (src/taxonomy.py), so printing reg_b_text for every listed
    # reason, not just the first, is what guarantees the grounding gate's
    # coverage check passes by construction.
    bullet_lines = "\n".join(f"- {r.reg_b_text}" for r in reasons)
    return (
        "After careful review of your application, we are unable to approve "
        "your request for credit at this time. The following factors, in "
        f"order of significance, contributed to this decision:\n{bullet_lines}\n"
        "You have the right to request the specific factors considered in "
        "this decision within 60 days."
    )


def llm_narrative(reasons: list[ReasonCode], tokenizer, model, feedback: str | None = None) -> str:
    """Prompt a local instruction-tuned causal LM to write a short,
    grounded adverse-action paragraph using ONLY the given reasons.

    `feedback` is populated on a regenerate-after-gate-failure attempt so
    the model sees exactly what it got wrong last time.
    """
    import torch

    allowed = "; ".join(r.reg_b_text for r in reasons)
    instruction = (
        "You are drafting one short paragraph (3-4 sentences) of an adverse "
        "action notice for a declined credit application. You MUST mention "
        f"each of these reasons, using natural language, and NO others: "
        f"{allowed}. Do not invent any reason not in this list."
    )
    if feedback:
        instruction += f" Correction needed: {feedback}"

    msgs = [{"role": "user", "content": instruction}]
    enc = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=120, do_sample=False)
    return tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


@dataclass
class GenerationResult:
    narrative: str
    gate: GateResult
    attempts: int
    backend: str  # "template" or "llm"
    fell_back_to_template: bool


def generate_with_gate(
    reasons: list[ReasonCode],
    use_llm: bool = False,
    tokenizer=None,
    model=None,
) -> GenerationResult:
    """Generate a narrative and run it through the deterministic grounding
    gate. If use_llm=True: try the LLM up to MAX_REGENERATE_ATTEMPTS times,
    feeding back exactly what the gate found wrong each time it fails; if
    it still fails after all attempts, fall back to the guaranteed-grounded
    template so a declined applicant never receives an ungrounded letter.
    """
    if not use_llm:
        narrative = template_narrative(reasons)
        gate = check_grounding(narrative, reasons)
        return GenerationResult(narrative, gate, attempts=1, backend="template", fell_back_to_template=False)

    feedback = None
    for attempt in range(1, MAX_REGENERATE_ATTEMPTS + 1):
        narrative = llm_narrative(reasons, tokenizer, model, feedback=feedback)
        gate = check_grounding(narrative, reasons)
        if gate.passed:
            return GenerationResult(narrative, gate, attempts=attempt, backend="llm", fell_back_to_template=False)
        parts = []
        if gate.missing_reasons:
            parts.append(f"you did not mention: {'; '.join(gate.missing_reasons)}")
        if gate.invented_reasons:
            parts.append(f"you incorrectly mentioned: {'; '.join(gate.invented_reasons)}")
        feedback = "; ".join(parts)

    # Exhausted retries: never ship an ungrounded letter, fall back.
    narrative = template_narrative(reasons)
    gate = check_grounding(narrative, reasons)
    return GenerationResult(
        narrative, gate, attempts=MAX_REGENERATE_ATTEMPTS, backend="template", fell_back_to_template=True
    )
