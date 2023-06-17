from __future__ import annotations

import numpy as np
import xgboost as xgb

from scratchboost import Booster


def test_results(X_y: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = X_y
    booster = Booster(
        iterations=30,
        learning_rate=0.3,
        max_depth=5,
        l2=1,
        min_leaf_weight=1,
        gamma=0,
        nbins=None,
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
        eval_metric="auc",
        # =500,
    )
    xgb_booster.fit(X, y)
    xpreds = xgb_booster.predict(X, output_margin=True)
    print(xgb_booster.get_booster().get_dump(with_stats=True)[0][0:200])
    assert np.allclose(bpreds, xpreds)
