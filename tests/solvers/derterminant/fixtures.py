from typing import Generator

import numpy as np


def data_provider_for_determinant() -> Generator[tuple[np.ndarray], None, None]:
    yield from [
        (
            np.array([
                [0, 2, 3, 4],
                [4, 0, 5, 1],
                [5, 6, 0, 8],
                [6, 5, 1, 0],
            ], dtype=np.float64),
        ),
        (
            np.array([
                [0, 2, 0, 0],
                [4, 0, 0, 0],
                [0, 0, 0, 8],
                [0, 0, 1, 0],
            ], dtype=np.float64),
        ),
        (
            np.array([
                [1, 2, 3, 0],
                [2, 4, 5, 0],
                [0, 0, 0, 1],
                [1, -4, 7, 0],
            ], dtype=np.float64),
        ),
    ]

    for i in range(100):
        n = np.random.randint(1, 8)
        zeros_count = np.random.randint(0, n*n)
        zero_rows, zero_cols = np.unravel_index(np.random.choice(n*n, zeros_count, replace=False), (n, n))

        matrix = np.random.random((n, n))
        matrix[zero_rows, zero_cols] = 0

        yield (matrix,)
