from __future__ import annotations
from scratchboost import Booster
import xgboost as xgb
import numpy as np

def test_results(X_y: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = X_y
    booster = Booster(
        iterations=30,
        learning_rate=0.3,
        max_depth=5,
        l2=1,
        min_leaf_weight=1,
        gamma=0,
    )
    booster.fit(X, y)
    bpreds = booster.predict(X)

    xgb_booster = xgb.XGBClassifier(
    n_estimators=30, 
    learning_rate=0.3,
    max_depth=5,
    reg_lambda=1,
    min_child_weight=1,
    gamma=0,
    objective="binary:logitraw",
    tree_method="hist",
    nbins=500,
    )
    xgb_booster.fit(X, y)
    xpreds = xgb_booster.predict(X, output_margin=True)
    assert np.allclose(bpreds, xpreds)



