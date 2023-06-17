from __future__ import annotations

from abc import ABC, abstractstaticmethod
from optparse import Option
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Type

import numpy as np
import numpy.typing as npt
import pandas as pd

from scratchboost.histogram import HistogramData
from scratchboost.node import TreeNode
from scratchboost.splitter import HistogramSplitter, Splitter
from scratchboost.utils import SplitInfo, bin_data, cover, gain, weight

# https://arxiv.org/pdf/1603.02754.pdf
# https://github.com/Ekeany/XGBoost-From-Scratch/blob/master/XGBoost.py
# https://medium.com/analytics-vidhya/what-makes-xgboost-so-extreme-e1544a4433bb


class LossABC(ABC):
    @abstractstaticmethod
    def loss(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:  # type: ignore
        return np.ndarray(())

    @abstractstaticmethod
    def grad(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:  # type: ignore
        return np.ndarray(())

    @abstractstaticmethod
    def hess(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:  # type: ignore
        return np.ndarray(())


class LogLoss(LossABC):
    """LogLoss, expects y_hat to be be a probability.
    Thus if it's on the logit scale, it will need to be converted
    to a probability using this function:
    y_hat = 1 / (1 + np.exp(-y_hat))
    """

    @staticmethod
    def loss(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        y_hat = 1 / (1 + np.exp(-y_hat))
        return -1 * (y * np.log(y_hat) + (1 - y) * (np.log(1 - y_hat)))

    @staticmethod
    def grad(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        y_hat = 1 / (1 + np.exp(-y_hat))
        return y_hat - y

    @staticmethod
    def hess(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        y_hat = 1 / (1 + np.exp(-y_hat))
        return y_hat * (1 - y_hat)


class Booster:
    def __init__(
        self,
        iterations: int = 10,
        objective: Type[LossABC] = LogLoss,
        l2: float = 1,
        gamma: float = 0,
        max_leaves: int = int(1e9),
        max_depth: int = 15,
        min_leaf_weight: float = 0,
        learning_rate: float = 0.3,
        base_score: float = 0.5,
        nbins: int | None = 250,
    ):
        self.obj = objective()
        self.iterations = iterations
        self.l2 = l2
        self.gamma = gamma
        self.min_leaf_weight = min_leaf_weight
        self.learning_rate = learning_rate
        self.max_leaves = max_leaves
        self.max_depth = max_depth
        self.base_score = base_score
        self.nbins = nbins
        self.trees_: List[Tree] = []
        self.splitter = HistogramSplitter(
            learning_rate=self.learning_rate,
            l2=self.l2,
            gamma=self.gamma,
            min_leaf_weight=min_leaf_weight,
        )

    def fit(
        self,
        X: npt.NDArray[np.float_],
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Booster:
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if np.any(np.isnan(X)) | np.any(np.isnan(y)):
            raise ValueError("Missing Values not supported, please impute them with a real value.")
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        if sample_weight is None:
            sample_weight_ = np.ones(y.shape)
        else:
            sample_weight_ = sample_weight
        preds_ = np.repeat(self.base_score, repeats=X.shape[0])
        gradient_ = self.obj.grad(y=y, y_hat=preds_) * sample_weight_
        hessian_ = self.obj.hess(y=y, y_hat=preds_) * sample_weight_
        X_binned, feature_cuts = bin_data(X, self.nbins)
        for _ in range(self.iterations):
            t = Tree(
                l2=self.l2,
                gamma=self.gamma,
                max_leaves=self.max_leaves,
                max_depth=self.max_depth,
                min_leaf_weight=self.min_leaf_weight,
                learning_rate=self.learning_rate,
            )
            self.trees_.append(
                t.fit(
                    X_binned=X_binned,
                    feature_cuts=feature_cuts,
                    gradient=gradient_,
                    hessian=hessian_,
                    splitter=self.splitter,
                )
            )
            preds_ += t.predict(X=X)
            gradient_ = self.obj.grad(y=y, y_hat=preds_) * sample_weight_
            hessian_ = self.obj.hess(y=y, y_hat=preds_) * sample_weight_
        return self

    def predict(self, X: npt.NDArray[np.float_]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if np.any(np.isnan(X)):
            raise ValueError("Missing Values not supported, please impute them with a real value.")
        preds_ = np.repeat(self.base_score, X.shape[0])
        for t in self.trees_:
            preds_ += t.predict(X)
        return preds_


class Tree:
    """
    Define a tree structure that houses a vector
    of nodes.
    """

    def __init__(
        self,
        l2: float = 1,
        gamma: float = 0,
        max_leaves: int = int(1e9),
        max_depth: int = 15,
        min_leaf_weight: float = 0,
        learning_rate: float = 0.3,
    ):
        self.l2 = l2
        self.gamma = gamma
        self.min_leaf_weight = min_leaf_weight
        self.learning_rate = learning_rate
        self.max_leaves = max_leaves
        self.max_depth = max_depth
        self.nodes_: List[TreeNode] = []

    def __repr__(self):
        r = ""
        print_buffer = [0]
        while len(print_buffer) > 0:
            n = self.nodes_[print_buffer.pop()]
            r += (n.depth * "      ") + n.__repr__() + "\n"
            if not n.is_leaf:
                print_buffer.append(n.right_child_)  # type: ignore
                print_buffer.append(n.left_child_)  # type: ignore
        return r

    def predict_row(self, x_row: np.ndarray) -> float:
        node_idx = 0
        while True:
            n = self.nodes_[node_idx]
            if n.is_leaf:
                return n.weight_value
            if x_row[n.split_feature_] < n.split_value_:  # type: ignore
                node_idx = n.left_child_
            else:
                node_idx = n.right_child_

    def predict(self, X: npt.NDArray[np.float_]) -> np.ndarray:
        preds_ = np.ndarray((X.shape[0],))
        for i in range(X.shape[0]):
            preds_[i] = self.predict_row(X[i, :])
        return preds_

    def fit(
        self,
        X_binned: npt.NDArray[np.int_],
        feature_cuts: list[npt.NDArray[np.float_]],
        gradient: np.ndarray,
        hessian: np.ndarray,
        splitter: HistogramSplitter,
    ) -> Tree:
        self.nodes_ = []
        gradient_sum = gradient.sum()
        hessian_sum = hessian.sum()
        root_gain = gain(gradient_sum=gradient_sum, hessian_sum=hessian_sum, l2=self.l2)
        root_weight = weight(
            gradient_sum=gradient_sum,
            hessian_sum=hessian_sum,
            l2=self.l2,
            learning_rate=self.learning_rate,
        )
        root_node = TreeNode(
            num=0,
            node_idxs=np.arange(X_binned.shape[0]),
            depth=0,
            weight_value=root_weight,
            gain_value=root_gain,
            cover_value=hessian_sum,
        )
        root_node.histograms_ = HistogramData.from_records(
            X=X_binned,
            feature_cuts=feature_cuts,
            gradient=gradient,
            hessian=hessian,
        )
        self.nodes_.append(root_node)
        n_leaves = 1
        growable = [0]
        while len(growable) > 0:
            if n_leaves >= self.max_leaves:
                break

            n_idx = growable.pop()

            n = self.nodes_[n_idx]
            depth = n.depth + 1

            # if we have hit max depth, skip this node
            # but keep going, because there may be other
            # valid shallower nodes.
            if depth > self.max_depth:
                continue

            # For max_leaves, subtract 1 from the n_leaves
            # everytime we pop from the growable stack
            # then, if we can add two children, add two to
            # n_leaves. If we can't split the node any
            # more, then just add 1 back to n_leaves
            n_leaves -= 1

            # Try to find a valid split for this node.
            split_info = splitter.best_split(
                node=n,
                gradient_sum=gradient[n.node_idxs].sum(),
                hessian_sum=hessian[n.node_idxs].sum(),
                feature_cuts=feature_cuts,
            )
            # If this is None, this means there
            # are no more valid nodes.
            if split_info is None:
                n_leaves += 1
                continue

            # If we can add two more leaves
            # add two.
            n_leaves += 2
            left_idx = len(self.nodes_)
            right_idx = left_idx + 1
            left_node = TreeNode(
                num=left_idx,
                node_idxs=split_info.left_idxs,
                weight_value=split_info.left_weight,
                gain_value=split_info.left_gain,
                cover_value=split_info.left_cover,
                depth=depth,
            )
            right_node = TreeNode(
                num=right_idx,
                node_idxs=split_info.right_idxs,
                weight_value=split_info.right_weight,
                gain_value=split_info.right_gain,
                cover_value=split_info.right_cover,
                depth=depth,
            )
            mask = (
                X_binned[n.node_idxs, split_info.split_feature] < split_info.split_index
            )
            left_node_idxs, right_node_idxs = n.node_idxs[mask], n.node_idxs[~mask]
            left_node.node_idxs = left_node_idxs
            right_node.node_idxs = right_node_idxs

            # Compute the histogram for the smalles node..
            if left_node_idxs.shape[0] < right_node_idxs.shape[0]:
                left_hist = HistogramData.from_records(
                    X_binned[left_node_idxs, :],
                    feature_cuts=feature_cuts,
                    gradient=gradient[left_node_idxs],
                    hessian=hessian[left_node_idxs],
                )
                right_hist = HistogramData.from_parent_child(n.histograms_, left_hist)
            else:
                right_hist = HistogramData.from_records(
                    X_binned[right_node_idxs, :],
                    feature_cuts=feature_cuts,
                    gradient=gradient[right_node_idxs],
                    hessian=hessian[right_node_idxs],
                )
                left_hist = HistogramData.from_parent_child(n.histograms_, right_hist)
            left_node.histograms_ = left_hist
            right_node.histograms_ = right_hist

            self.nodes_.append(left_node)
            self.nodes_.append(right_node)
            # Get indexes
            n.update_children(
                left_child=left_idx, right_child=right_idx, split_info=split_info
            )
            growable.insert(0, left_idx)
            growable.insert(0, right_idx)

        return self


# def split_gain(
#     left_mask: np.ndarray,
#     right_mask: np.ndarray,
#     gradient: np.ndarray,
#     hessian: np.ndarray,
#     l2: float,
#     gamma: float,
# ) -> float:
#     gl = gradient[left_mask].sum()
#     gr = gradient[right_mask].sum()
#     hl = hessian[left_mask].sum()
#     hr = hessian[right_mask].sum()
#     l = (gl**2) / (hl + l2)
#     r = (gr**2) / (hr + l2)
#     lr = ((gl + gr) ** 2) / (hl + hr + l2)
#     return (l + r - lr) - gamma


# def missing_gain(
#     left_mask: np.ndarray,
#     right_mask: np.ndarray,
#     missing_mask: np.ndarray,
#     gradient: np.ndarray,
#     hessian: np.ndarray,
#     l2: float,
#     gamma: float,
# ) -> Tuple[float, float]:
#     gl = gradient[left_mask].sum()
#     gr = gradient[right_mask].sum()
#     gm = gradient[missing_mask].sum()
#     hl = hessian[left_mask].sum()
#     hr = hessian[right_mask].sum()
#     hm = hessian[missing_mask].sum()
#     l = (gl**2) / (hl + l2)
#     lm = ((gl + gm) ** 2) / (hl + hm + l2)
#     r = (gr**2) / (hr + l2)
#     rm = ((gr + gm) ** 2) / (hr + hm + l2)
#     lrm = ((gl + gr + gm) ** 2) / (hl + hr + hm + l2)
#     gain_left = (lm + r - lrm) - gamma
#     gain_right = (l + rm - lrm) - gamma
#     return (gain_left, gain_right)


# def weight(
#     gradient: np.ndarray,
#     hessian: np.ndarray,
#     l2: float,
# ) -> float:
#     return -1 * (gradient.sum() / (hessian.sum() + l2))
