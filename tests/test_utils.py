from __future__ import annotations
import numpy as np
from scratchboost.utils import bin_data
import pytest

@pytest.mark.parametrize("nbins", range(25, 300, 25))
def test_binning(nbins: int, X_y: tuple[np.ndarray, np.ndarray]) -> None:
    X, _ = X_y
    b, c = bin_data(X, nbins)

    for i in range(X.shape[1]):
        assert all(X[:,i] < c[i][b[:,i]])