"""Per-applicant SHAP attribution -> fixed-taxonomy reason codes.

This is the step that turns a black-box LightGBM prediction into the
specific, individualized "principal reasons" Regulation B requires. It is
pure numeric SHAP math plus a dictionary lookup -- no LLM involved -- so
it is deterministic and auditable on its own, independent of anything the
generation step does downstream.
"""

from dataclasses import dataclass

import lightgbm as lgb
import pandas as pd
import shap

from src.taxonomy import PROTECTED_FEATURES, ReasonCode, reason_for


@dataclass
class ApplicantReasons:
    index: int
    reasons: list[ReasonCode]  # top-K, ordered by |SHAP value|, protected features excluded
    shap_values: dict  # feature -> raw shap value, for the audit trail


def build_explainer(model: lgb.LGBMClassifier) -> shap.TreeExplainer:
    return shap.TreeExplainer(model)


def top_reasons_for_applicant(
    explainer: shap.TreeExplainer,
    X_row: pd.Series,
    k: int = 3,
) -> ApplicantReasons:
    """Return the top-K SHAP-driven reasons pushing this applicant toward
    decline, restricted to features that are (a) actually pushing toward
    the adverse outcome in the taxonomy's declared direction and (b) not
    a protected/excluded feature.
    """
    row_df = X_row.to_frame().T
    sv = explainer.shap_values(row_df)
    # shap_values for a binary LGBMClassifier TreeExplainer returns either
    # a (1, n_features) array for the positive class, or a list of two
    # such arrays ([class0, class1]); normalize to the positive-class row.
    if isinstance(sv, list):
        sv = sv[1]
    sv_row = sv[0]

    contributions = dict(zip(X_row.index, sv_row))

    candidates = []
    for feature, value in contributions.items():
        if feature in PROTECTED_FEATURES:
            continue
        rc = reason_for(feature)
        if rc is None:
            continue
        # Only count it as a *reason for decline* if the SHAP value pushes
        # toward the positive (high-risk/decline) class, i.e. value > 0.
        # Direction sanity check (high features should push positive when
        # value is high, etc.) is left to the model; here we only require
        # the SHAP contribution itself to be adverse (positive).
        if value > 0:
            candidates.append((feature, value, rc))

    candidates.sort(key=lambda t: t[1], reverse=True)

    # Two GMSC features (30-59 and 60-89 days late) map to the same
    # canonical ReasonCode (see src/taxonomy.py) -- de-duplicate by
    # identity so a single Reg B reason is never listed twice.
    top: list[ReasonCode] = []
    seen_ids = set()
    for _, _, rc in candidates:
        if id(rc) in seen_ids:
            continue
        seen_ids.add(id(rc))
        top.append(rc)
        if len(top) == k:
            break

    return ApplicantReasons(
        index=int(X_row.name),
        reasons=top,
        shap_values={f: float(v) for f, v in contributions.items()},
    )
