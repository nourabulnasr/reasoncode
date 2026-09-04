"""Train the LightGBM credit-risk classifier used as the decision engine.

The model's predicted class stands in for the credit decision: predicted
1 (high risk of serious delinquency) => decline => adverse action notice
required. This mirrors how a real risk model gates a real underwriting
decision; GMSC's label is the closest public proxy available for that.
"""

from dataclasses import dataclass

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.data import FEATURES, TARGET, load_gmsc


@dataclass
class TrainedModel:
    model: lgb.LGBMClassifier
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    test_auc: float


def train(seed: int = 42) -> TrainedModel:
    df = load_gmsc()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    clf = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=seed,
        verbose=-1,
    )
    clf.fit(X_train, y_train)
    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    return TrainedModel(clf, X_train, X_test, y_train, y_test, float(auc))


def declined_mask(model: lgb.LGBMClassifier, X: pd.DataFrame) -> pd.Series:
    """Predicted-positive (high delinquency risk) rows == declined applicants."""
    return pd.Series(model.predict(X), index=X.index).astype(bool)
