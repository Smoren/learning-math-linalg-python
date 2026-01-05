from typing import Generator

import numpy as np


def data_provider_for_mul_polynomials() -> Generator[tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
    yield from [
        (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        ),
        (
            np.array([1], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        ),
        (
            np.array([1, 2], dtype=np.float64),
            np.array([1], dtype=np.float64),
            np.array([1, 2], dtype=np.float64),
        ),
        (
            np.array([0, 0, 1], dtype=np.float64),
            np.array([1], dtype=np.float64),
            np.array([0, 0, 1], dtype=np.float64),
        ),
        (
            np.array([1, 2], dtype=np.float64),
            np.array([2], dtype=np.float64),
            np.array([2, 4], dtype=np.float64),
        ),
        (
            np.array([2], dtype=np.float64),
            np.array([1, 2], dtype=np.float64),
            np.array([2, 4], dtype=np.float64),
        ),
        (
            np.array([1, 2], dtype=np.float64),
            np.array([3, 4], dtype=np.float64),
            np.array([3, 10, 8], dtype=np.float64),
        ),
        (
            np.array([1, 2, 3], dtype=np.float64),
            np.array([10, 100], dtype=np.float64),
            np.array([10, 120, 230, 300], dtype=np.float64),
        ),
        (
            np.array([0, 0, 1], dtype=np.float64),
            np.array([10, 100], dtype=np.float64),
            np.array([0, 0, 10, 100], dtype=np.float64),
        ),
        (
            np.array([0, 1], dtype=np.float64),
            np.array([10, 100], dtype=np.float64),
            np.array([0, 10, 100], dtype=np.float64),
        ),
        (
            np.array([1, 2, 3], dtype=np.float64),
            np.array([4, 5, 6], dtype=np.float64),
            np.array([4, 13, 28, 27, 18], dtype=np.float64),
        ),
    ]
