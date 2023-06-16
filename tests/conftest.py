from __future__ import annotations
import seaborn
import pytest
import numpy as np

@pytest.fixture(scope="session")
def X_y() -> tuple[np.ndarray, np.ndarray]:
    df = seaborn.load_dataset("titanic")
    X = df.select_dtypes("number").fillna(0).drop(columns="survived").to_numpy()
    y = df["survived"]
    return X, y
