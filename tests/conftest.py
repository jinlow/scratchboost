from __future__ import annotations

import numpy as np
import pytest
import seaborn


@pytest.fixture(scope="session")
def X_y() -> tuple[np.ndarray, np.ndarray]:
    df = seaborn.load_dataset("titanic")
    X = df.select_dtypes("number").fillna(0).drop(columns="survived").to_numpy()
    y = df["survived"]
    return X, y
