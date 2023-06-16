from __future__ import annotations
import numpy as np
import numpy.typing as npt


def create_histogram(
    feature: npt.NDArray[np.int_],
    gradient: npt.NDArray[np.float32],
    hessian: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    gradient_bins = np.bincount(feature, weights=gradient).astype(np.float32)
    hessian_bins = np.bincount(feature, weights=hessian).astype(np.float32)
    return gradient_bins, hessian_bins


class HistogramData:
    def __init__(
        self,
    ) -> None:
        self.histograms = []

    @classmethod
    def from_records(
        cls,
        X: npt.NDArray[np.int_],
        gradient: npt.NDArray[np.float32],
        hessian: npt.NDArray[np.float32],
    ) -> HistogramData:
        hd = cls()
        for i in range(X.shape[1]):
            hd.histograms.append(
                create_histogram(X[:, i], gradient=gradient, hessian=hessian)
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
