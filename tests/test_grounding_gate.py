from src.grounding_gate import check_grounding
from src.taxonomy import TAXONOMY

REVOLVING = TAXONOMY["RevolvingUtilizationOfUnsecuredLines"]
DELINQUENCY_90 = TAXONOMY["NumberOfTimes90DaysLate"]
DEBT_RATIO = TAXONOMY["DebtRatio"]


def test_passes_when_all_allowed_reasons_are_mentioned_and_nothing_else():
    narrative = f"Declined due to: {REVOLVING.reg_b_text}. {DELINQUENCY_90.reg_b_text}."
    result = check_grounding(narrative, [REVOLVING, DELINQUENCY_90])
    assert result.passed
    assert result.missing_reasons == []
    assert result.invented_reasons == []


def test_fails_on_missing_required_reason():
    narrative = f"Declined due to: {REVOLVING.reg_b_text}."
    result = check_grounding(narrative, [REVOLVING, DELINQUENCY_90])
    assert not result.passed
    assert result.missing_reasons == [DELINQUENCY_90.reg_b_text]


def test_fails_on_invented_reason_not_in_allowed_set():
    # Model was only allowed to talk about revolving utilization, but the
    # narrative also brings up debt ratio -- an invented/hallucinated reason.
    narrative = f"Declined due to: {REVOLVING.reg_b_text}. Also, {DEBT_RATIO.reg_b_text}."
    result = check_grounding(narrative, [REVOLVING])
    assert not result.passed
    assert result.invented_reasons == [DEBT_RATIO.reg_b_text]


def test_case_insensitive_matching():
    narrative = REVOLVING.reg_b_text.upper()
    result = check_grounding(narrative, [REVOLVING])
    assert result.passed
