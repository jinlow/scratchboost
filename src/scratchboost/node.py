import numpy as np

from scratchboost.histogram import HistogramData
from scratchboost.utils import SplitInfo


class TreeNode:
    """Node of the tree, this determines the split, or if this
    is a terminal node value.
    """

    split_feature_: float
    split_gain_: float
    histograms_: HistogramData

    children_: list[int]
    children_bounds_: list[tuple[float, float]]

    def __init__(
        self,
        num: int,
        weight_value: float,
        gain_value: float,
        cover_value: float,
        depth: int,
    ):
        self.num = num

        self.weight_value = weight_value
        self.gain_value = gain_value
        self.cover_value = cover_value
        self.depth = depth

    def prep_for_split(self, node_idxs: np.ndarray, histograms: HistogramData) -> None:
        self.node_idxs = node_idxs
        self.histograms_ = histograms

    def __repr__(self) -> str:
        if self.is_leaf:
            n = (
                f"{repr(self.num)}:leaf={repr(self.weight_value)},"
                + f"cover={repr(self.cover_value)}"
            )
        else:
            n = (
                f"{repr(self.num)}:[{repr(self.split_feature_)} < "
                + f"{repr(self.children_bounds_[0][1])}] yes={repr(self.children_[0])},"
                + f"no={repr(self.children_[1])},"
                + f"gain={repr(self.split_gain_)},cover={repr(self.cover_value)}"
            )
        return n

    def print_node(self):
        return (
            "TreeNode{\n"
            + f"\tweight_value: {self.weight_value}\n"
            + f"\tgain_value: {self.gain_value}\n"
            + f"\tcover_value: {self.cover_value}\n"
            + f"\tdepth: {self.depth}\n"
            + f"\tsplit_value_: {self.children_bounds_[0][1]}\n"
            + f"\tsplit_feature_: {self.split_feature_}\n"
            + f"\tsplit_gain_: {self.split_gain_}\n"
            + f"\tleft_child_: {self.children_[0]}\n"
            + f"\tright_child_: {self.children_[1]}\n"
            + "        }"
        )

    @property
    def is_leaf(self):
        return not hasattr(self, "children_")

    def get_next_node(self, value: float):
        # We can have a binary tree, or a ternary tree.
        second_bounds = self.children_bounds_[1]
        if value < second_bounds[0]:
            return self.children_[0]
        if value < second_bounds[1]:
            return self.children_[1]
        else:
            return self.children_[2]

    def add_children(
        self,
        children: list[int],
        split_info: SplitInfo,
    ):
        self.children_ = children
        self.children_bounds_ = split_info.children_bounds
        self.split_gain_ = np.sum(split_info.children_gain)
        self.split_feature_ = split_info.split_feature
        # Clear histograms and index references
        if hasattr(self, "histograms_"):
            del self.histograms_
        del self.node_idxs

    # def update_children(
    #     self,
    #     left_child: int,
    #     right_child: int,
    #     split_info: SplitInfo,
    # ):
    #     """
    #     Update the children, and split information for the node.
    #     """
    #     self.left_child_ = left_child
    #     self.right_child_ = right_child
    #     self.split_feature_ = split_info.split_feature
    #     self.split_gain_ = (
    #         split_info.left_gain + split_info.right_gain - self.gain_value
    #     )
    #     self.split_value_ = split_info.split_value[0]
