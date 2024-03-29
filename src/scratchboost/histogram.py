from __future__ import annotations

import numpy as np
import numpy.typing as npt
import functools


def create_histogram(
    feature: npt.NDArray[np.int_],
    feature_len: int,
    gradient: npt.NDArray[np.float32],
    hessian: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    gradient_bins = np.bincount(
        feature, weights=gradient, minlength=feature_len
    ).astype(np.float32)
    hessian_bins = np.bincount(feature, weights=hessian, minlength=feature_len).astype(
        np.float32
    )
    return gradient_bins, hessian_bins


class HistogramData:
    def __init__(
        self,
    ) -> None:
        self.histograms = []

    def __getitem__(self, v: int) -> tuple[np.ndarray, np.ndarray]:
        return self.histograms[v]

    @classmethod
    def from_records(
        cls,
        X: npt.NDArray[np.int_],
        feature_cuts: list[npt.NDArray[np.float_]],
        gradient: npt.NDArray[np.float32],
        hessian: npt.NDArray[np.float32],
    ) -> HistogramData:
        hd = cls()
        for i in range(X.shape[1]):
            hd.histograms.append(
                create_histogram(
                    X[:, i],
                    feature_len=feature_cuts[i].shape[0],
                    gradient=gradient,
                    hessian=hessian,
                )
            )
        return hd

    @classmethod
    def from_parent_child(
        cls,
        parent: HistogramData,
        child: HistogramData,
    ) -> HistogramData:
        hd = cls()
        for p, c in zip(parent.histograms, child.histograms):
            hd.histograms.append((p[0] - c[0], p[1] - c[1]))
        return hd

    @classmethod
    def from_parent_children(
        cls,
        parent: HistogramData,
        *children: HistogramData,
    ):
        hd = cls()
        children_ = functools.reduce(
            lambda x, y: (x[0] + y[0], x[1] + y[1]),
            (c.histograms for c in children),
        )
        for p, c in zip(parent.histograms, children_):
            hd.histograms.append((p[0] - c[0], p[1] - c[1]))
        return hd
