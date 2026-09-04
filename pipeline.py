"""ReasonCode end-to-end pipeline: train -> explain -> generate -> audit.

Run `python pipeline.py demo` for the 30-second path (see README.md).
"""

import argparse
import json
import sys

from src.data import sentinel_flag_count, load_gmsc
from src.explain import build_explainer, top_reasons_for_applicant
from src.fairness import four_fifths_audit
from src.generate import generate_with_gate
from src.model import declined_mask, train


def run_demo(sample_size: int = 20, top_k: int = 3):
    print(f"[1/4] Loading Give Me Some Credit and training LightGBM classifier...")
    df = load_gmsc()
    print(f"      {len(df)} rows loaded ({sentinel_flag_count(df)} rows have a known GMSC late-payment sentinel value, kept as-is)")
    tm = train()
    print(f"      test-set AUC: {tm.test_auc:.4f}")

    print(f"\n[2/4] Scoring the test set and finding declined applicants (predicted high-risk)...")
    declined = declined_mask(tm.model, tm.X_test)
    n_declined = int(declined.sum())
    print(f"      {n_declined} of {len(tm.X_test)} test applicants declined by the model")

    sample_idx = tm.X_test[declined].index[:sample_size]
    print(f"      generating grounded reason-code letters for a fixed sample of {len(sample_idx)} declined applicants")

    print(f"\n[3/4] SHAP attribution -> fixed taxonomy -> grounded narrative + gate check...")
    explainer = build_explainer(tm.model)
    results = []
    gate_pass_count = 0
    for idx in sample_idx:
        row = tm.X_test.loc[idx]
        ar = top_reasons_for_applicant(explainer, row, k=top_k)
        gen = generate_with_gate(ar.reasons, use_llm=False)
        gate_pass_count += int(gen.gate.passed)
        results.append(
            {
                "applicant_index": int(idx),
                "reasons": [r.reg_b_text for r in ar.reasons],
                "narrative": gen.narrative,
                "gate_passed": gen.gate.passed,
                "backend": gen.backend,
            }
        )
    print(f"      grounding gate: {gate_pass_count}/{len(results)} narratives passed on first generation")
    print(f"\n      --- sample letter (applicant {results[0]['applicant_index']}) ---")
    print(f"      reasons: {results[0]['reasons']}")
    print(f"      {results[0]['narrative']}")

    print(f"\n[4/4] Fairness audit (four-fifths rule, full test set, age groups split at 40)...")
    fr = four_fifths_audit(declined.astype(int), tm.X_test["age"])
    print(f"      decline rates by group: {fr.group_decline_rates}")
    print(f"      four-fifths ratio: {fr.four_fifths_ratio:.3f} -> {'PASSES' if fr.passes_four_fifths_rule else 'FAILS'} the 80% rule")

    return {
        "test_auc": tm.test_auc,
        "n_declined_sampled": len(results),
        "gate_pass_rate": gate_pass_count / len(results),
        "fairness": {
            "group_decline_rates": fr.group_decline_rates,
            "four_fifths_ratio": fr.four_fifths_ratio,
            "passes_four_fifths_rule": fr.passes_four_fifths_rule,
        },
        "sample_letters": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["demo"])
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--out", type=str, default=None, help="optional path to dump full JSON results")
    args = parser.parse_args()

    if args.command == "demo":
        result = run_demo(sample_size=args.sample_size)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\nFull results written to {args.out}")
