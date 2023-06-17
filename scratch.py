import seaborn as sns
import xgboost as xgb
from scratchboost.model import Booster
import numpy as np

df = sns.load_dataset("titanic")
X = df.select_dtypes("number").fillna(0).drop(columns="survived").to_numpy()
y = df["survived"]

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
    eval_metric="auc",
)
xgb_booster.fit(X, y)
xpreds = xgb_booster.predict(X, output_margin=True)
np.allclose(bpreds, xpreds)
