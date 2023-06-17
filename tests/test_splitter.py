import numpy as np

from scratchboost.model import LogLoss
from scratchboost.node import TreeNode
from scratchboost.splitter import ExactSplitter
from scratchboost.utils import gain, weight


def test_exact(X_y: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = X_y
    splitter = ExactSplitter(
        learning_rate=0.3,
        min_leaf_weight=1,
        gamma=0,
        l2=1,
    )
    preds = np.repeat(0.5, y.shape)
    gradient = LogLoss().grad(y, preds)
    hessian = LogLoss().hess(y, preds)
    gradient_sum = gradient.sum()
    hessian_sum = hessian.sum()
    root_gain = gain(gradient_sum=gradient_sum, hessian_sum=hessian_sum, l2=splitter.l2)
    root_weight = weight(
        gradient_sum=gradient_sum,
        hessian_sum=hessian_sum,
        l2=splitter.l2,
        learning_rate=splitter.learning_rate,
    )
    root_node = TreeNode(
        num=0,
        node_idxs=np.arange(X.shape[0]),
        depth=0,
        weight_value=root_weight,
        gain_value=root_gain,
        cover_value=hessian_sum,
    )

    info = splitter.best_feature_split(
        root_node, X, 2, gradient=gradient, hessian=hessian
    )
