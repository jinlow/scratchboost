from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class SplitInfo:
    split_gain: float
    split_feature: int
    split_value: list[float]
    split_index: int

    children_gain: list[float]
    children_cover: list[float]
    children_weight: list[float]
    children_bounds: list[tuple[float, float]]


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


def get_cuts(f: npt.NDArray[np.float_], percentiles: npt.NDArray[np.float_] | None):
    if percentiles is None:
        vals = np.sort(np.unique(f))[1:]
    else:
        vals = np.unique(np.percentile(f, percentiles, method="midpoint"))[1:]
    return np.append(vals, np.inf)


def bin_data(
    X: npt.NDArray[np.float_], nbins: int | None
) -> tuple[npt.NDArray[np.int_], list[npt.NDArray[np.float_]]]:
    XB = np.empty(X.shape, dtype=np.int32)
    cuts_: list[npt.NDArray[np.float_]] = []
    percentiles = np.linspace(0, 100, num=nbins + 1) if nbins is not None else None
    for i in range(X.shape[1]):
        f = X[:, i]
        f.max()
        p = get_cuts(X[:, i], percentiles)
        XB[:, i] = np.digitize(f, p, right=False)
        cuts_.append(p)
    return XB, cuts_
