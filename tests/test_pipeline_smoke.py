"""End-to-end smoke test: real GMSC data, real LightGBM model, real SHAP
attribution, real grounding gate -- no mocks. Trains once per test session
(module-scoped fixture) since training + SHAP setup take real wall time.
"""

import pytest

from src.explain import build_explainer, top_reasons_for_applicant
from src.generate import generate_with_gate
from src.model import declined_mask, train


@pytest.fixture(scope="module")
def trained():
    return train()


@pytest.fixture(scope="module")
def explainer(trained):
    return build_explainer(trained.model)


def test_model_beats_chance(trained):
    # CLAIM_UNDER_TEST for the underlying classifier itself: it must be
    # meaningfully better than a coin flip (0.5 AUC) on held-out data.
    assert trained.test_auc > 0.75


def test_declined_applicants_get_at_least_one_reason(trained, explainer):
    declined = declined_mask(trained.model, trained.X_test)
    sample = trained.X_test[declined].index[:10]
    assert len(sample) > 0, "expected at least some declined applicants in the test set"
    for idx in sample:
        ar = top_reasons_for_applicant(explainer, trained.X_test.loc[idx], k=3)
        assert len(ar.reasons) >= 1, f"applicant {idx} was declined but got zero explainable reasons"


def test_protected_features_never_appear_as_reasons(trained, explainer):
    declined = declined_mask(trained.model, trained.X_test)
    sample = trained.X_test[declined].index[:30]
    for idx in sample:
        ar = top_reasons_for_applicant(explainer, trained.X_test.loc[idx], k=3)
        for r in ar.reasons:
            assert "age" not in r.reg_b_text.lower()
            assert "dependent" not in r.reg_b_text.lower()


def test_template_generation_always_passes_gate(trained, explainer):
    declined = declined_mask(trained.model, trained.X_test)
    sample = trained.X_test[declined].index[:30]
    for idx in sample:
        ar = top_reasons_for_applicant(explainer, trained.X_test.loc[idx], k=3)
        gen = generate_with_gate(ar.reasons, use_llm=False)
        assert gen.gate.passed, f"applicant {idx}: {gen.gate.missing_reasons} {gen.gate.invented_reasons}"
