from __future__ import annotations

import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from typing import Any

@dataclass
class SplitInfo:
    split_gain: float
    split_feature: int
    split_value: Any
    split_index: int

    left_gain: float
    left_cover: float
    left_weight: float
    left_idxs: np.ndarray

    right_gain: float
    right_cover: float
    right_weight: float
    right_idxs: np.ndarray

def cover(
    hessian: np.ndarray,
) -> float:
    return hessian.sum()


def gain(
    gradient_sum: float,
    hessian_sum: float,
    l2: float,
) -> float:
    return (gradient_sum**2) / (hessian_sum + l2)


def weight(
    gradient_sum: float, hessian_sum: float, l2: float, learning_rate: float
) -> float:
    return -1 * (gradient_sum / (hessian_sum + l2)) * learning_rate


def bin_data(
    X: npt.NDArray[np.float_], nbins: int
) -> tuple[npt.NDArray[np.int_], list[npt.NDArray[np.float_]]]:
    XB = np.empty(X.shape, dtype=np.int32)
    cuts_: list[npt.NDArray[np.float_]] = []
    for i in range(X.shape[1]):
        f = X[:, i]
        max_ = f.max()
        p = np.append(
            np.unique(np.quantile(X[:, i], np.linspace(0, 1, num=nbins))), np.inf
        )
        XB[:, i] = np.digitize(f, p, right=False)
        cuts_.append(p)
    return XB, cuts_
