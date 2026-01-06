from typing import Generator

import numpy as np


def data_provider_for_add_polynomials() -> Generator[tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
    yield from [
        (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([0], dtype=np.float64),
        ),
        (
            np.array([0], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([0], dtype=np.float64),
        ),
        (
            np.array([0], dtype=np.float64),
            np.array([0], dtype=np.float64),
            np.array([0], dtype=np.float64),
        ),
        (
            np.array([1], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([1], dtype=np.float64),
        ),
        (
            np.array([], dtype=np.float64),
            np.array([1, 2], dtype=np.float64),
            np.array([1, 2], dtype=np.float64),
        ),
        (
            np.array([1, 2, 3], dtype=np.float64),
            np.array([4, 5, 6], dtype=np.float64),
            np.array([5, 7, 9], dtype=np.float64),
        ),
        (
            np.array([1, 2], dtype=np.float64),
            np.array([4, 5, 6], dtype=np.float64),
            np.array([5, 7, 6], dtype=np.float64),
        ),
        (
            np.array([1, 2, 3], dtype=np.float64),
            np.array([4, 5], dtype=np.float64),
            np.array([5, 7, 3], dtype=np.float64),
        ),
        (
            np.array([1, 2, 3], dtype=np.float64),
            np.array([4, 5, -3], dtype=np.float64),
            np.array([5, 7], dtype=np.float64),
        ),
        (
            np.array([1, 2, 3], dtype=np.float64),
            np.array([0, 0, 0, 4, 5, 6], dtype=np.float64),
            np.array([1, 2, 3, 4, 5, 6], dtype=np.float64),
        ),
        (
            np.array([0, 0, 0, 4, 5, 6], dtype=np.float64),
            np.array([1, 2, 3, -4, -5, -6], dtype=np.float64),
            np.array([1, 2, 3], dtype=np.float64),
        ),
        (
            np.array([0, 0, 0, 4, 5, 6], dtype=np.float64),
            np.array([1, 2, 3, -4, -5, -6, -7], dtype=np.float64),
            np.array([1, 2, 3, 0, 0, 0, -7], dtype=np.float64),
        ),
    ]
