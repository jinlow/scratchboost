import numpy as np

import typing

from scratchboost.histogram import HistogramData
from scratchboost.utils import SplitInfo


class TreeNode:
    """Node of the tree, this determines the split, or if this
    is a terminal node value.
    """

    split_value_: float
    split_feature_: float
    split_gain_: float
    left_child_: int
    right_child_: int
    histograms_: HistogramData

    def __init__(
        self,
        num: int,
        node_idxs: np.ndarray,
        weight_value: float,
        gain_value: float,
        cover_value: float,
        depth: int,
    ):
        self.num = num
        self.node_idxs = node_idxs

        self.weight_value = weight_value
        self.gain_value = gain_value
        self.cover_value = cover_value
        self.depth = depth

    def __repr__(self) -> str:
        if self.is_leaf:
            n = f"{repr(self.num)}:leaf={repr(self.weight_value)},cover={repr(self.cover_value)}"
        else:
            n = f"{repr(self.num)}:[{repr(self.split_feature_)} < {repr(self.split_value_)}] yes={repr(self.left_child_)},no={repr(self.right_child_)},gain={repr(self.split_gain_)},cover={repr(self.cover_value)}"
        return n

    def print_node(self):
        return (
            "TreeNode{\n"
            + f"\tweight_value: {self.weight_value}\n"
            + f"\tgain_value: {self.gain_value}\n"
            + f"\tcover_value: {self.cover_value}\n"
            + f"\tdepth: {self.depth}\n"
            + f"\tsplit_value_: {self.split_value_}\n"
            + f"\tsplit_feature_: {self.split_feature_}\n"
            + f"\tsplit_gain_: {self.split_gain_}\n"
            + f"\tleft_child_: {self.left_child_}\n"
            + f"\tright_child_: {self.right_child_}\n"
            + "        }"
        )

    @property
    def is_leaf(self):
        return hasattr(self, "split_feature_")

    def get_next_node(self, value: float):
        if value < self.split_value_:
            return self.left_child_
        else:
            return self.right_child_

    def update_children(
        self,
        left_child: int,
        right_child: int,
        split_info: SplitInfo,
    ):
        """
        Update the children, and split information for the node.
        """
        self.left_child_ = left_child
        self.right_child_ = right_child
        self.split_feature_ = split_info.split_feature
        self.split_gain_ = (
            split_info.left_gain + split_info.right_gain - self.gain_value
        )
        self.split_value_ = split_info.split_value
        # Clear histograms and index references
        if hasattr(self, "histograms_"):
            del self.histograms_
        del self.node_idxs
