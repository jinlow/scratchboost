from __future__ import annotations
import numpy as np
import numpy.typing as npt

def bin_data(X: npt.NDArray[np.float_], nbins: int) -> tuple[npt.NDArray[np.int_], list[npt.NDArray[np.float_]]]:
    XB = np.empty(X.shape, dtype=np.int32)
    cuts_: list[npt.NDArray[np.float_]] = []
    for i in range(X.shape[1]):
        f = X[:,i]
        max_ = f.max()
        p = np.append(np.unique(np.quantile(X[:,i], np.linspace(0, 1, num=nbins))), np.inf)
        XB[:,i] = np.digitize(f, p, right=False)
        cuts_.append(p)
    return XB, cuts_