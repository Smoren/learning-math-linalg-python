from typing import Generator

import numpy as np


def data_provider_for_add_matrices() -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    yield from [
        (
            np.array([
                [1, 2],
                [3, 4],
            ], dtype=np.float64),
            np.array([
                [1, 2],
                [3, 4],
            ], dtype=np.float64),
        ),
    ]


def data_provider_for_mul_matrices() -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    yield from [
        (
            np.array([
                [1, 2],
                [3, 4],
            ], dtype=np.float64),
            np.array([
                [1, 2],
                [3, 4],
            ], dtype=np.float64),
        ),
        (
            np.array([
                [1, 2],
                [4, 5],
                [7, 8],
            ], dtype=np.float64),
            np.array([
                [5, 6, 7, 8],
                [7, 8, 9, 10],
            ], dtype=np.float64),
        ),
    ]
