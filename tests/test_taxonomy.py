"""Guards the keyword-uniqueness property the grounding gate depends on.

This is a regression test for the real bug found during the first demo
run (see NOTES.md / git history): keywords that weren't literal
substrings of their own reg_b_text broke coverage; keywords that were
substrings of each other's text would break the no-invention check the
same way. Both are asserted here so neither can silently regress.
"""

from src.taxonomy import CANONICAL_REASONS, PROTECTED_FEATURES, TAXONOMY


def test_every_keyword_is_substring_of_its_own_reg_b_text():
    for r in CANONICAL_REASONS:
        text = r.reg_b_text.lower()
        for kw in r.keywords:
            assert kw in text, f"keyword {kw!r} is not a substring of its own reg_b_text {r.reg_b_text!r}"


def test_no_two_distinct_reasons_share_a_cross_matching_keyword():
    for i, a in enumerate(CANONICAL_REASONS):
        for b in CANONICAL_REASONS[i + 1:]:
            for kw_a in a.keywords:
                for kw_b in b.keywords:
                    assert kw_a not in kw_b and kw_b not in kw_a, (
                        f"keyword overlap between distinct reasons: {kw_a!r} <-> {kw_b!r}"
                    )


def test_protected_features_are_not_in_taxonomy():
    for pf in PROTECTED_FEATURES:
        assert pf not in TAXONOMY


def test_day_bucket_features_share_canonical_reason():
    assert TAXONOMY["NumberOfTime30-59DaysPastDueNotWorse"] is TAXONOMY["NumberOfTime60-89DaysPastDueNotWorse"]
