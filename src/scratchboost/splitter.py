from __future__ import annotations

import abc
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from scratchboost.node import TreeNode
from scratchboost.utils import SplitInfo, gain, weight


class Splitter(abc.ABC):
    def __init__(
        self,
        learning_rate: float,
        min_leaf_weight: float,
        l2: float,
        gamma: float,
    ) -> None:
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.l2 = l2
        self.min_leaf_weight = min_leaf_weight

    def best_split(
        self,
        node: TreeNode,
        X: npt.NDArray[np.int_],
        gradient: np.ndarray,
        hessian: np.ndarray,
    ) -> Optional[SplitInfo]:
        pass

    def best_feature_split(
        self,
        node: TreeNode,
        X: npt.NDArray[np.int_],
        feature: int,
        gradient: np.ndarray,
        hessian: np.ndarray,
    ) -> Optional[SplitInfo]:
        pass


class HistogramSplitter(Splitter):
    def best_split(
        self,
        node: TreeNode,
        gradient_sum: np.ndarray,
        hessian_sum: np.ndarray,
        feature_cuts: list[npt.NDArray[np.float_]],
    ) -> Optional[SplitInfo]:

        # Split info
        best_gain = -np.inf
        best_split_info = None

        for f in range(len(feature_cuts)):
            split_info = self.best_feature_split(
                node=node,
                feature=f,
                gradient_sum=gradient_sum,
                hessian_sum=hessian_sum,
                cuts=feature_cuts[f],
            )

            if split_info is None:
                continue

            if split_info.split_gain > best_gain:
                best_gain = split_info.split_gain
                best_split_info = split_info
        return best_split_info

    def best_feature_split(
        self,
        node: TreeNode,
        feature: int,
        gradient_sum: np.ndarray,
        hessian_sum: np.ndarray,
        cuts: npt.NDArray[np.float_],
    ) -> SplitInfo | None:
        max_gain = -np.inf
        split_info = None
        g, h = node.histograms_[feature]
        left_gradient = left_hessian = 0.0
        for i in range(0, g.shape[0] - 1):
            left_gradient += g[i]
            left_hessian += h[i]
            right_gradient, right_hessian = (
                gradient_sum - left_gradient,
                hessian_sum - left_hessian,
            )
            if np.min([left_hessian, right_hessian]) < self.min_leaf_weight:
                continue
            left_gain = gain(
                gradient_sum=left_gradient, hessian_sum=left_hessian, l2=self.l2
            )
            right_gain = gain(
                gradient_sum=right_gradient, hessian_sum=right_hessian, l2=self.l2
            )
            split_gain = (left_gain + right_gain - node.gain_value) - self.gamma
            if split_gain <= 0:
                continue
            if split_gain > max_gain:
                max_gain = split_gain
                split_info = SplitInfo(
                    split_gain=split_gain,
                    split_feature=feature,
                    split_value=cuts[i],
                    split_index=i + 1,
                    left_gain=left_gain,
                    left_cover=left_hessian,
                    left_weight=weight(
                        gradient_sum=left_gradient,
                        hessian_sum=left_hessian,
                        l2=self.l2,
                        learning_rate=self.learning_rate,
                    ),
                    left_idxs=np.empty((0, 0)),
                    right_gain=right_gain,
                    right_cover=right_hessian,
                    right_weight=weight(
                        gradient_sum=right_gradient,
                        hessian_sum=right_hessian,
                        l2=self.l2,
                        learning_rate=self.learning_rate,
                    ),
                    right_idxs=np.empty((0, 0)),
                )
        return split_info


class ExactSplitter(Splitter):
    def best_split(
        self,
        node: TreeNode,
        X: npt.NDArray[np.int_],
        gradient: np.ndarray,
        hessian: np.ndarray,
    ) -> Optional[SplitInfo]:
        """
        Find the best split for this node out of all the features.
        """
        X_ = X[node.node_idxs, :]
        gradient_ = gradient[node.node_idxs]
        hessian_ = hessian[node.node_idxs]

        # Split info
        best_gain = -np.inf
        best_split_info = None

        for f in range(X_.shape[1]):
            split_info = self.best_feature_split(
                node=node,
                X=X_,
                feature=f,
                gradient=gradient_,
                hessian=hessian_,
            )

            if split_info is None:
                continue

            if split_info.split_gain > best_gain:
                best_gain = split_info.split_gain
                best_split_info = split_info
        return best_split_info

    def best_feature_split(
        self,
        node: TreeNode,
        X: npt.NDArray[np.int_],
        feature: int,
        gradient: np.ndarray,
        hessian: np.ndarray,
    ) -> Optional[SplitInfo]:
        """
        Find the best split for a given feature, if it is
        possible to create a split with this feature.
        """
        max_gain = -np.inf
        split_info = None

        # Skip the first value, because nothing is smaller
        # than the first value.
        x = X[:, feature]
        split_vals = np.unique(x)
        for v in split_vals[1:]:
            mask = x < v
            lidxs, ridxs = node.node_idxs[mask], node.node_idxs[~mask]  # type: ignore
            lgs, lhs = gradient[mask].sum(), hessian[mask].sum()
            rgs, rhs = gradient[~mask].sum(), hessian[~mask].sum()
            # Don't even consider this if the min_leaf_weight
            # parameter is violated.
            if np.min([lhs, rhs]) < self.min_leaf_weight:
                continue
            l_gain = gain(lgs, lhs, l2=self.l2)
            r_gain = gain(rgs, rhs, l2=self.l2)
            split_gain = (l_gain + r_gain - node.gain_value) - self.gamma
            if split_gain <= 0:
                continue
            if split_gain > max_gain:
                max_gain = split_gain
                split_info = SplitInfo(
                    split_index=0,
                    split_gain=split_gain,
                    split_feature=feature,
                    split_value=v,
                    left_gain=l_gain,
                    left_cover=lhs,
                    left_weight=weight(
                        gradient_sum=lgs,
                        hessian_sum=lhs,
                        l2=self.l2,
                        learning_rate=self.learning_rate,
                    ),
                    left_idxs=lidxs,
                    right_gain=r_gain,
                    right_cover=rhs,
                    right_weight=weight(
                        gradient_sum=rgs,
                        hessian_sum=rhs,
                        l2=self.l2,
                        learning_rate=self.learning_rate,
                    ),
                    right_idxs=ridxs,
                )
        return split_info
