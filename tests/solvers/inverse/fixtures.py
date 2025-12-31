from typing import Generator

import numpy as np


def data_provider_for_inverse_matrix() -> Generator[tuple[np.ndarray], None, None]:
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
